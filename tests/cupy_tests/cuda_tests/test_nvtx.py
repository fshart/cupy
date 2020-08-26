import unittest

from cupy import cuda
from cupy import testing


@unittest.skipUnless(cuda.nvtx.available, 'nvtx is not installed')
class TestNVTX(unittest.TestCase):

    @testing.gpu
    def test_Mark(self):
        cuda.nvtx.Mark('test:Mark', 0)

    @testing.gpu
    def test_MarkC(self):
        cuda.nvtx.MarkC('test:MarkC', 0xFF000000)

    @testing.gpu
    def test_RangePush(self):
        cuda.nvtx.RangePush('test:RangePush', 1)
        cuda.nvtx.RangePop()

    @testing.gpu
    def test_RangePushC(self):
        cuda.nvtx.RangePushC('test:RangePushC', 0xFF000000)
        cuda.nvtx.RangePop()
