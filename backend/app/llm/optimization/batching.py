"""
Batch processing for LLM requests.
"""
from typing import Dict, Any, List, Optional, Union, TypeVar, Generic, Callable
import asyncio
from app.utils.logger import get_logger

logger = get_logger("llm.optimization.batching")

T = TypeVar('T')
R = TypeVar('R')

class BatchProcessor(Generic[T, R]):
    """
    Batch processor for efficient processing of multiple requests.
    
    This class collects items and processes them in batches when the batch size is reached
    or when a timeout occurs.
    """
    
    def __init__(
        self, 
        batch_size: int, 
        timeout: float,
        processor_fn: Callable[[List[T]], asyncio.Future[List[R]]]
    ):
        """
        Initialize the batch processor.
        
        Args:
            batch_size: Maximum number of items per batch.
            timeout: Maximum time to wait for batch completion (seconds).
            processor_fn: Function to process a batch of items.
        """
        self.batch_size = batch_size
        self.timeout = timeout
        self.processor_fn = processor_fn
        self.batch: List[T] = []
        self.futures: List[asyncio.Future] = []
        self.batch_future: Optional[asyncio.Future] = None
        self.timer: Optional[asyncio.TimerHandle] = None
        self.lock = asyncio.Lock()
        
        logger.debug(f"Initialized BatchProcessor with batch_size={batch_size}, timeout={timeout}")
    
    async def process(self, item: T) -> R:
        """
        Add an item to the batch and get a future for its result.
        
        Args:
            item: Item to process.
            
        Returns:
            Result of processing the item.
        """
        future = asyncio.Future()
        
        async with self.lock:
            # Add item and future to batch
            self.batch.append(item)
            self.futures.append(future)
            
            logger.debug(f"Added item to batch ({len(self.batch)}/{self.batch_size})")
            
            # Start timer if this is the first item
            if len(self.batch) == 1:
                self._start_timer()
            
            # Process batch if it's full
            if len(self.batch) >= self.batch_size:
                await self._process_batch()
        
        # Wait for result
        return await future
    
    def _start_timer(self):
        """Start the timeout timer for batch processing."""
        if self.timer:
            self.timer.cancel()
        
        loop = asyncio.get_running_loop()
        self.timer = loop.call_later(
            self.timeout, 
            lambda: asyncio.create_task(self._process_batch_on_timeout())
        )
        
        logger.debug(f"Started batch timer with timeout={self.timeout}s")
    
    async def _process_batch_on_timeout(self):
        """Process the current batch if the timeout is reached."""
        async with self.lock:
            if self.batch:
                logger.debug("Processing batch due to timeout")
                await self._process_batch()
    
    async def _process_batch(self):
        """Process the current batch of items."""
        if not self.batch:
            return
        
        # Cancel timer if active
        if self.timer:
            self.timer.cancel()
            self.timer = None
        
        # Get current batch and futures
        current_batch = self.batch
        current_futures = self.futures
        
        # Clear batch and futures for next round
        self.batch = []
        self.futures = []
        
        logger.debug(f"Processing batch of {len(current_batch)} items")
        
        try:
            # Process batch
            results = await self.processor_fn(current_batch)
            
            # Set results for futures
            for future, result in zip(current_futures, results):
                if not future.done():
                    future.set_result(result)
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            
            # Set exception for all futures
            for future in current_futures:
                if not future.done():
                    future.set_exception(e)