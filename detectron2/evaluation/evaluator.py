# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(
    cfg, args, model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("---- Use NHWC format")
    if args.nv_fuser:
        fuser_mode = "fuser2"
    else:
       fuser_mode = "none"
    print("---- fuser mode:", fuser_mode)

    total_time = 0.0
    total_sample = 0

    if args.profile and args.device == "xpu":
        with ExitStack() as stack:
            if isinstance(model, nn.Module):
                stack.enter_context(inference_context(model))
            stack.enter_context(torch.no_grad())

            profile_len = len(data_loader) // 2
            for i, inputs in enumerate(data_loader):
                if i > args.num_iter:
                    break
                if args.jit and i == 0:
                    try:
                        model = torch.jit.trace(model, inputs, check_trace=False)
                        print("---- JIT trace enable.")
                    except (RuntimeError, TypeError) as e:
                        print("---- JIT trace disable.")
                        print("failed to use PyTorch jit mode due to: ", e)

                with torch.autograd.profiler_legacy.profile(enabled=args.profile, use_xpu=True, record_shapes=False) as prof:
                    start_time = time.time()
                    outputs = model(inputs)
                    torch.xpu.synchronize()
                    elapsed = time.time() - start_time
                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                    total_sample += args.batch_size
                    total_time += elapsed
                if args.profile and i == profile_len:
                    import pathlib
                    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                    if not os.path.exists(timeline_dir):
                        try:
                            os.makedirs(timeline_dir)
                        except:
                            pass
                    torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"),
                        timeline_dir+'profile.pt')
                    torch.save(prof.key_averages(group_by_input_shape=True).table(),
                        timeline_dir+'profile_detail.pt')
                    prof.export_chrome_trace("model_profile.json")
    elif args.profile and args.device == "cuda":
        with ExitStack() as stack:
            if isinstance(model, nn.Module):
                stack.enter_context(inference_context(model))
            stack.enter_context(torch.no_grad())
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                schedule=torch.profiler.schedule(
                    wait=profile_len,
                    warmup=2,
                    active=1,
                ),
                on_trace_ready=trace_handler,
            ) as p:
                for i, inputs in enumerate(data_loader):
                    if i > args.num_iter:
                        break
                    if args.jit and i == 0:
                        try:
                            model = torch.jit.trace(model, inputs, check_trace=False)
                            print("---- JIT trace enable.")
                        except (RuntimeError, TypeError) as e:
                            print("---- JIT trace disable.")
                            print("failed to use PyTorch jit mode due to: ", e)

                    start_time = time.time()
                    with torch.jit.fuser(fuser_mode):
                        outputs = model(inputs)
                    torch.cuda.synchronize()
                    elapsed = time.time() - start_time
                    p.step()
                    print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                    if i >= args.num_warmup:
                        total_sample += args.batch_size
                        total_time += elapsed
    elif args.profile and args.device == "cpu":
        with ExitStack() as stack:
            if isinstance(model, nn.Module):
                stack.enter_context(inference_context(model))
            stack.enter_context(torch.no_grad())
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,
                schedule=torch.profiler.schedule(
                    wait=profile_len,
                    warmup=2,
                    active=1,
                ),
                on_trace_ready=trace_handler,
            ) as p:
                for i, inputs in enumerate(data_loader):
                    if i > args.num_iter:
                        break
                    if args.jit and i == 0:
                        try:
                            model = torch.jit.trace(model, inputs, check_trace=False)
                            print("---- JIT trace enable.")
                        except (RuntimeError, TypeError) as e:
                            print("---- JIT trace disable.")
                            print("failed to use PyTorch jit mode due to: ", e)

                    start_time = time.time()
                    outputs = model(inputs)
                    elapsed = time.time() - start_time
                    p.step()
                    print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                    if i >= args.num_warmup:
                        total_sample += args.batch_size
                        total_time += elapsed
    elif not args.profile and args.device == "cuda":
        with ExitStack() as stack:
            if isinstance(model, nn.Module):
                stack.enter_context(inference_context(model))
            stack.enter_context(torch.no_grad())

            for i, inputs in enumerate(data_loader):
                if i > args.num_iter:
                    break
                if args.jit and i == 0:
                    try:
                        model = torch.jit.trace(model, inputs, check_trace=False)
                        print("---- JIT trace enable.")
                    except (RuntimeError, TypeError) as e:
                        print("---- JIT trace disable.")
                        print("failed to use PyTorch jit mode due to: ", e)

                start_time = time.time()
                with torch.jit.fuser(fuser_mode):
                    outputs = model(inputs)
                torch.cuda.synchronize()
                elapsed = time.time() - start_time
                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                    total_sample += args.batch_size
                    total_time += elapsed
    else:
        with ExitStack() as stack:
            if isinstance(model, nn.Module):
                stack.enter_context(inference_context(model))
            stack.enter_context(torch.no_grad())

            for i, inputs in enumerate(data_loader):
                if i > args.num_iter:
                    break
                #if args.channels_last:
                #    inputs = inputs.to(memory_format=torch.channels_last) if len(inputs.size()) == 4 else inputs
                if args.jit and i == 0:
                    try:
                        model = torch.jit.trace(model, inputs, check_trace=False)
                        print("---- JIT trace enable.")
                    except (RuntimeError, TypeError) as e:
                        print("---- JIT trace disable.")
                        print("failed to use PyTorch jit mode due to: ", e)

                start_time = time.time()
                outputs = model(inputs)
                if args.device == "xpu":
                    torch.xpu.synchronize()
                elapsed = time.time() - start_time
                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                    total_sample += args.batch_size
                    total_time += elapsed

    latency = total_time / total_sample * 1000
    throughput = total_sample / total_time
    print("inference Latency: {} ms".format(latency))
    print("inference Throughput: {} samples/s".format(throughput))

    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
