# mixins.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu
"""
Mixin classes for pynapse core objects.

This module provides mixin classes that add functionality to core data classes
like Sample and Population. The TensorConfigMixin enables integrated tensor
extraction with caching and default parameter support.
"""

from typing import Any, Dict, List, Optional, Union


class TensorConfigMixin:
    """Mixin providing tensor extraction and caching for Sample/Population.

    This mixin adds the ability to configure default tensor parameters and
    cache extracted tensors for efficient repeated access. Classes using this
    mixin must implement `create_tensor()` to define how tensors are created.

    Attributes:
        _tensor_cache: Dictionary caching tensors keyed by parameter tuples.
        _default_tensor_params: Default parameters for tensor extraction.
    """

    _tensor_cache: Dict[tuple, Any]
    _default_tensor_params: Dict[str, Any]

    def _init_tensor_config(
        self,
        default_event_id: Optional[Union[int, List[int]]] = None,
        default_pre_event: Optional[float] = None,
        default_post_event: Optional[float] = None,
        default_buffer_ms: int = 0,
        default_min_trials: int = 1,
    ) -> None:
        """Initialize tensor configuration and cache.

        Args:
            default_event_id: Default event ID(s) for tensor extraction.
            default_pre_event: Default pre-event window in seconds.
            default_post_event: Default post-event window in seconds.
            default_buffer_ms: Default minimum interval between events (ms).
            default_min_trials: Default minimum number of valid trials.
        """
        self._tensor_cache = {}
        self._default_tensor_params = {
            "event_id": default_event_id,
            "pre_event": default_pre_event,
            "post_event": default_post_event,
            "buffer_ms": default_buffer_ms,
            "min_trials": default_min_trials,
        }

    def _make_cache_key(
        self,
        event_id: Union[int, List[int]],
        pre_event: float,
        post_event: float,
        buffer_ms: int,
        min_trials: int,
        trace_preprocess: Any,
        window_preprocess: Any,
    ) -> tuple:
        """Create a hashable cache key from tensor parameters.

        Args:
            event_id: Event ID(s) for extraction.
            pre_event: Pre-event window in seconds.
            post_event: Post-event window in seconds.
            buffer_ms: Minimum interval between events (ms).
            min_trials: Minimum number of valid trials.
            trace_preprocess: Preprocessor for full trace (uses id() for key).
            window_preprocess: Preprocessor for windows (uses id() for key).

        Returns:
            Tuple that can be used as a dictionary key.
        """
        event_id_key = tuple(event_id) if isinstance(event_id, list) else (event_id,)
        return (
            event_id_key,
            pre_event,
            post_event,
            buffer_ms,
            min_trials,
            id(trace_preprocess) if trace_preprocess is not None else None,
            id(window_preprocess) if window_preprocess is not None else None,
        )

    def create_tensor(
        self,
        event_id: Union[int, List[int]],
        pre_event: float,
        post_event: float,
        buffer_ms: int,
        min_trials: int,
        trace_preprocess: Any,
        window_preprocess: Any,
    ) -> Any:
        """Create a tensor object. Must be implemented by subclasses.

        Args:
            event_id: Event ID(s) for extraction.
            pre_event: Pre-event window in seconds.
            post_event: Post-event window in seconds.
            buffer_ms: Minimum interval between events (ms).
            min_trials: Minimum number of valid trials.
            trace_preprocess: Preprocessor for full trace.
            window_preprocess: Preprocessor for windows.

        Returns:
            The created tensor object (SampleEventTensor or PopulationEventTensor).

        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError("Subclasses must implement create_tensor()")

    def get_tensor(
        self,
        event_id: Optional[Union[int, List[int]]] = None,
        pre_event: Optional[float] = None,
        post_event: Optional[float] = None,
        buffer_ms: Optional[int] = None,
        min_trials: Optional[int] = None,
        trace_preprocess: Any = None,
        window_preprocess: Any = None,
        use_cache: bool = True,
    ) -> Any:
        """Get or create a tensor for peri-event analysis.

        Parameters not provided will fall back to defaults set during
        construction. If a required parameter (event_id, pre_event, post_event)
        has no default and is not provided, a ValueError is raised.

        Args:
            event_id: Event ID(s) to extract windows around.
            pre_event: Seconds before event to include.
            post_event: Seconds after event to include.
            buffer_ms: Minimum interval between valid events (ms).
            min_trials: Minimum number of valid trials per sample.
            trace_preprocess: Preprocessor/Pipeline for full trace.
            window_preprocess: Preprocessor/Pipeline for extracted windows.
            use_cache: If True, return cached tensor if available.

        Returns:
            SampleEventTensor or PopulationEventTensor depending on class.

        Raises:
            ValueError: If required parameters are missing.
        """
        resolved_event_id = event_id if event_id is not None else self._default_tensor_params["event_id"]
        resolved_pre_event = pre_event if pre_event is not None else self._default_tensor_params["pre_event"]
        resolved_post_event = post_event if post_event is not None else self._default_tensor_params["post_event"]
        resolved_buffer_ms = buffer_ms if buffer_ms is not None else self._default_tensor_params["buffer_ms"]
        resolved_min_trials = min_trials if min_trials is not None else self._default_tensor_params["min_trials"]

        if resolved_event_id is None:
            raise ValueError("event_id must be provided or set as default")
        if resolved_pre_event is None:
            raise ValueError("pre_event must be provided or set as default")
        if resolved_post_event is None:
            raise ValueError("post_event must be provided or set as default")

        cache_key = self._make_cache_key(
            resolved_event_id,
            resolved_pre_event,
            resolved_post_event,
            resolved_buffer_ms,
            resolved_min_trials,
            trace_preprocess,
            window_preprocess,
        )

        if use_cache and cache_key in self._tensor_cache:
            return self._tensor_cache[cache_key]

        tensor = self.create_tensor(
            resolved_event_id,
            resolved_pre_event,
            resolved_post_event,
            resolved_buffer_ms,
            resolved_min_trials,
            trace_preprocess,
            window_preprocess,
        )

        if use_cache:
            self._tensor_cache[cache_key] = tensor

        return tensor

    def clear_tensor_cache(self) -> None:
        """Clear all cached tensors to free memory."""
        self._tensor_cache.clear()
