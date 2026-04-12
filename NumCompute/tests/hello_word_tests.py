import unittest


def say_hello_word():
    return "hello word"


class TestHelloWord(unittest.TestCase):
    def test_hello_word_message(self):
        self.assertEqual(say_hello_word(), "hello word")


if __name__ == "__main__":
    unittest.main()
