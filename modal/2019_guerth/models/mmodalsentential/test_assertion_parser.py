import unittest
from assertion_parser import parse_one, parse_all, Expr

class TestParse(unittest.TestCase):

    def test_parse_one(self):
        tests = [
            (
                'a',
                Expr(op='id', args=['a'])
            ),
            (
                '  a    ',
                Expr(op='id', args=['a'])
            ),
            (
                '  A    small    Fox ',
                Expr(op='id', args=['A small Fox'])
            ),
            (
                'if a or b then c',
                Expr(op='id', args=['if a or b then c'])
            ),
            (
                'a | b',
                Expr(op='|', args=[Expr(op='id', args=['a']), Expr(op='id', args=['b'])])
            ),
            (
                'a|b',
                Expr(op='|', args=[Expr(op='id', args=['a']), Expr(op='id', args=['b'])])
            ),
            (
                'Hallo | Servus ',
                Expr(op='|', args=[Expr(op='id', args=['Hallo']), Expr(op='id', args=['Servus'])])
            ),
            (
                'a | b & c -> []d',
                Expr(op='->', args=[Expr(op='|', args=[Expr(op='id', args=['a']), Expr(op='&', args=[Expr(op='id', args=['b']), Expr(op='id', args=['c'])])]), Expr(op='[]', args=[Expr(op='id', args=['d'])])])
            ),
            (
                'a & b | c & d',
                Expr(op='|', args=[Expr(op='&', args=[Expr(op='id', args=['a']), Expr(op='id', args=['b'])]), Expr(op='&', args=[Expr(op='id', args=['c']), Expr(op='id', args=['d'])])])
            ),
            (
                '~[]<>~d',
                Expr(op='~', args=[Expr(op='[]', args=[Expr(op='<>', args=[Expr(op='~', args=[Expr(op='id', args=['d'])])])])])
            ),
            (
                'a | b -> c',
                Expr(op='->', args=[Expr(op='|', args=[Expr(op='id', args=['a']), Expr(op='id', args=['b'])]), Expr(op='id', args=['c'])])
            ),
            (
                'God exists | atheism is right',
                Expr(op='|', args=[Expr(op='id', args=['God exists']), Expr(op='id', args=['atheism is right'])])
            ),
            (
                '<>a & <>b',
                Expr(op='&', args=[Expr(op='<>', args=[Expr(op='id', args=['a'])]), Expr(op='<>', args=[Expr(op='id', args=['b'])])])
            )
        ]
        for assertion, expected in tests:
            with self.subTest(value=assertion):
                self.assertEqual(parse_one(assertion), expected)

        # Testing Exceptions:
        with self.subTest(value='a || b'):
            with self.assertRaises(Exception):
                parse_one('a || b')
        with self.subTest(value='((a)'):
            with self.assertRaises(Exception):
                parse_one('((a)')
        with self.subTest(value=' '):
            with self.assertRaises(Exception):
                parse_one(' ')
        with self.subTest(value=''):
            with self.assertRaises(Exception):
                parse_one('')

    def test_parse_all(self):
        tests = [
            (
                [
                    "a | b -> c",
                    'a'
                ],
                [
                    Expr(op='->', args=[Expr(op='|', args=[Expr(op='id', args=['a']), Expr(op='id', args=['b'])]), Expr(op='id', args=['c'])]),
                    Expr(op='id', args=['a'])
                ]
            ),
            (
                [
                    '<>a & <>b',
                    '<>(a & b)'
                ],
                [
                    Expr(op='&', args=[Expr(op='<>', args=[Expr(op='id', args=['a'])]), Expr(op='<>', args=[Expr(op='id', args=['b'])])]),
                    Expr(op='<>', args=[Expr(op='&', args=[Expr(op='id', args=['a']), Expr(op='id', args=['b'])])])
                ]
            ),
            (
                [
                    '<>a & <>b',
                    '<>(a & b)',
                    'c'
                ],
                [
                    Expr(op='&', args=[Expr(op='<>', args=[Expr(op='id', args=['a'])]), Expr(op='<>', args=[Expr(op='id', args=['b'])])]),
                    Expr(op='<>', args=[Expr(op='&', args=[Expr(op='id', args=['a']), Expr(op='id', args=['b'])])]),
                    Expr(op='id', args=['c'])
                ]
            ),
            (
                [],
                []
            )
        ]
        for assertion, expected in tests:
            with self.subTest(value=assertion):
                self.assertEqual(parse_all(assertion), expected)

if __name__ == '__main__':
    unittest.main()