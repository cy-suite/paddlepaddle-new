# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from paddle.jit.sot.infer_meta import rehash_int


class TestRehash(unittest.TestCase):
    def test_rehash_int(self):
        # -1 and -2 should have different hash values
        # If we use hash directly, they will have the same hash value.
        # It may cause cache collision.
        self.assertNotEqual(hash(rehash_int(-1)), hash(rehash_int(-2)))

    def test_rehash_int_same(self):
        self.assertEqual(rehash_int(1), rehash_int(1))

    def test_rehash_bool(self):
        self.assertNotEqual(rehash_int(True), rehash_int(1))
        self.assertNotEqual(rehash_int(False), rehash_int(0))


if __name__ == "__main__":
    unittest.main()
