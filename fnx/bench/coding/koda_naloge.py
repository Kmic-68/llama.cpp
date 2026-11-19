#!/usr/bin/env python3
"""koda_naloge.py (2026-11-18, q4hitri): coding tasks with hidden tests for judging whether a lossy speed-up hurts coding.

FUNKCIJE: (id, prompt, tests, ref) - write one function or class; the answer's longest ```python block is run against
the asserts. ref is our own solution, only for preveri_naloge.py (it must pass, so the tests are right).
UREJANJE: Koda-style edits. mode "orodje": the model sees the file through a read tool result and must change it with
edit/write tool calls (oldString copied exactly). mode "cela": the model returns the whole changed file in one block.
Each has the original file, the instruction, tests (run on the edited file) and ref_edit (old, new) pairs that turn
the original into a correct answer (preveri_naloge.py: the original fails the tests, the ref passes).

The tasks are our own (written for this test), so no model saw them in training. Mixed English and Slovenian like Koda's
users. Imported by koda_ab.py (the round's ab.py) and by preveri_naloge.py."""

FUNKCIJE = [
("spiral",
 "Write a Python function `spiral_order(matrix: list[list[int]]) -> list[int]` that returns the elements of a rectangular matrix in clockwise spiral order, starting at the top-left corner.",
 r'''assert spiral_order([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]
assert spiral_order([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) == [1,2,3,4,8,12,11,10,9,5,6,7]
assert spiral_order([[1],[2],[3]]) == [1,2,3]
assert spiral_order([[1,2,3]]) == [1,2,3]
assert spiral_order([]) == []''',
 r'''def spiral_order(m):
    res = []
    if not m:
        return res
    top, bot, left, right = 0, len(m) - 1, 0, len(m[0]) - 1
    while top <= bot and left <= right:
        for j in range(left, right + 1):
            res.append(m[top][j])
        top += 1
        for i in range(top, bot + 1):
            res.append(m[i][right])
        right -= 1
        if top <= bot:
            for j in range(right, left - 1, -1):
                res.append(m[bot][j])
            bot -= 1
        if left <= right:
            for i in range(bot, top - 1, -1):
                res.append(m[i][left])
            left += 1
    return res'''),

("decode",
 "Write a Python function `decode_string(s: str) -> str` that decodes strings like '3[a2[c]]' -> 'accaccacc': k[x] repeats x k times (k is a positive integer, possibly with several digits), brackets nest, and letters outside brackets stay as they are.",
 r'''assert decode_string('3[a]2[bc]') == 'aaabcbc'
assert decode_string('3[a2[c]]') == 'accaccacc'
assert decode_string('2[abc]3[cd]ef') == 'abcabccdcdcdef'
assert decode_string('10[x]') == 'x' * 10
assert decode_string('ab') == 'ab'
assert decode_string('2[a2[b3[c]]]d') == 'abcccbcccabcccbcccd' ''',
 r'''def decode_string(s):
    stack, cur, num = [], '', 0
    for ch in s:
        if ch.isdigit():
            num = num * 10 + int(ch)
        elif ch == '[':
            stack.append((cur, num))
            cur, num = '', 0
        elif ch == ']':
            prev, k = stack.pop()
            cur = prev + cur * k
        else:
            cur += ch
    return cur'''),

("nextperm",
 "Write a Python function `next_permutation(nums: list[int]) -> list[int]` that returns, as a new list, the next lexicographically greater permutation of nums, or the lowest one (sorted ascending) if nums is already the greatest. nums may contain duplicates. Do not modify the input list.",
 r'''a = [1, 2, 3]
assert next_permutation(a) == [1, 3, 2] and a == [1, 2, 3]
assert next_permutation([3, 2, 1]) == [1, 2, 3]
assert next_permutation([1, 1, 5]) == [1, 5, 1]
assert next_permutation([1, 5, 1]) == [5, 1, 1]
assert next_permutation([2, 3, 1]) == [3, 1, 2]
assert next_permutation([1]) == [1]
assert next_permutation([]) == []''',
 r'''def next_permutation(nums):
    a = list(nums)
    i = len(a) - 2
    while i >= 0 and a[i] >= a[i + 1]:
        i -= 1
    if i >= 0:
        j = len(a) - 1
        while a[j] <= a[i]:
            j -= 1
        a[i], a[j] = a[j], a[i]
    a[i + 1:] = reversed(a[i + 1:])
    return a'''),

("palindrom",
 "Write a Python function `longest_palindrome(s: str) -> str` that returns the longest palindromic substring of s. If several have the same length, return the one that starts first; return '' for ''.",
 r'''assert longest_palindrome('babad') == 'bab'
assert longest_palindrome('cbbd') == 'bb'
assert longest_palindrome('a') == 'a'
assert longest_palindrome('') == ''
assert longest_palindrome('forgeeksskeegfor') == 'geeksskeeg'
assert longest_palindrome('abacdfgdcaba') == 'aba'
assert longest_palindrome('aaaa') == 'aaaa' ''',
 r'''def longest_palindrome(s):
    best = ''
    for c in range(len(s)):
        for l, r in ((c, c), (c, c + 1)):
            while l >= 0 and r < len(s) and s[l] == s[r]:
                l -= 1
                r += 1
            cand = s[l + 1:r]
            if len(cand) > len(best) or (len(cand) == len(best) and cand and s.find(cand) < s.find(best)):
                best = cand
    return best'''),

("pot",
 "Write a Python function `simplify_path(path: str) -> str` that simplifies an absolute Unix path: collapse repeated slashes, drop '.', let '..' go up one directory (never above '/'), no trailing slash except for the root. Names like '...' are ordinary directory names.",
 r'''assert simplify_path('/home/') == '/home'
assert simplify_path('/../') == '/'
assert simplify_path('/home//foo/') == '/home/foo'
assert simplify_path('/a/./b/../../c/') == '/c'
assert simplify_path('/.../a') == '/.../a'
assert simplify_path('/a//b////c/d//././/..') == '/a/b/c' ''',
 r'''def simplify_path(path):
    stack = []
    for part in path.split('/'):
        if part in ('', '.'):
            continue
        if part == '..':
            if stack:
                stack.pop()
        else:
            stack.append(part)
    return '/' + '/'.join(stack)'''),

("besede_en",
 "Write a Python function `number_to_words(n: int) -> str` for 0 <= n < 2**31 that writes n in English words, each word capitalized, without 'and' and without hyphens: 0 -> 'Zero', 123 -> 'One Hundred Twenty Three', 1000010 -> 'One Million Ten'.",
 r'''assert number_to_words(0) == 'Zero'
assert number_to_words(13) == 'Thirteen'
assert number_to_words(20) == 'Twenty'
assert number_to_words(100) == 'One Hundred'
assert number_to_words(123) == 'One Hundred Twenty Three'
assert number_to_words(12345) == 'Twelve Thousand Three Hundred Forty Five'
assert number_to_words(1000000) == 'One Million'
assert number_to_words(1000010) == 'One Million Ten'
assert number_to_words(1234567) == 'One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven'
assert number_to_words(2147483647) == 'Two Billion One Hundred Forty Seven Million Four Hundred Eighty Three Thousand Six Hundred Forty Seven' ''',
 r'''def number_to_words(n):
    ones = 'Zero One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen'.split()
    tens = 'Zero Ten Twenty Thirty Forty Fifty Sixty Seventy Eighty Ninety'.split()
    def three(x):
        w = []
        if x >= 100:
            w += [ones[x // 100], 'Hundred']
            x %= 100
        if x >= 20:
            w.append(tens[x // 10])
            x %= 10
        if x:
            w.append(ones[x])
        return w
    if n == 0:
        return 'Zero'
    words = []
    for value, name in ((10**9, 'Billion'), (10**6, 'Million'), (1000, 'Thousand'), (1, '')):
        if n >= value:
            words += three(n // value) + ([name] if name else [])
            n %= value
    return ' '.join(words)'''),

("racunalo",
 "Write a Python function `calc(expr: str) -> int` that evaluates an integer expression with + - * /, parentheses, spaces and unary minus (also right after an operator or '('). * and / bind tighter than + and -, all four are left-associative, and / is integer division truncating toward zero (as in C). Do not use eval.",
 r'''assert calc('1 + 1') == 2
assert calc(' 2-1 + 2 ') == 3
assert calc('(1+(4+5+2)-3)+(6+8)') == 23
assert calc('-(2+3)*4') == -20
assert calc('14/-3') == -4
assert calc('2*(3+4)-10/3') == 11
assert calc('7-3-2') == 2
assert calc('100/10/5') == 2
assert calc('-7/2') == -3
assert calc('2*-3') == -6''',
 r'''def calc(expr):
    s = expr.replace(' ', '')
    pos = 0
    def peek():
        return s[pos] if pos < len(s) else ''
    def expr_():
        nonlocal pos
        v = term()
        while peek() in ('+', '-') and peek():
            op = s[pos]; pos += 1
            r = term()
            v = v + r if op == '+' else v - r
        return v
    def term():
        nonlocal pos
        v = unary()
        while peek() in ('*', '/') and peek():
            op = s[pos]; pos += 1
            r = unary()
            if op == '*':
                v = v * r
            else:
                q = abs(v) // abs(r)
                v = q if (v >= 0) == (r >= 0) else -q
        return v
    def unary():
        nonlocal pos
        if peek() == '-':
            pos += 1
            return -unary()
        if peek() == '(':
            pos += 1
            v = expr_()
            pos += 1
            return v
        start = pos
        while peek().isdigit():
            pos += 1
        return int(s[start:pos])
    return expr_()'''),

("zvezdica",
 "Write a Python function `wildcard_match(s: str, p: str) -> bool` that says whether the pattern p matches the whole string s, where '?' matches any one character and '*' matches any sequence of characters (also an empty one).",
 r'''assert not wildcard_match('aa', 'a')
assert wildcard_match('aa', '*')
assert not wildcard_match('cb', '?a')
assert wildcard_match('adceb', '*a*b')
assert not wildcard_match('acdcb', 'a*c?b')
assert wildcard_match('', '*')
assert not wildcard_match('', '?')
assert wildcard_match('abcabczzzde', '*abc???de*')''',
 r'''def wildcard_match(s, p):
    dp = [[False] * (len(p) + 1) for _ in range(len(s) + 1)]
    dp[0][0] = True
    for j in range(1, len(p) + 1):
        dp[0][j] = dp[0][j - 1] and p[j - 1] == '*'
    for i in range(1, len(s) + 1):
        for j in range(1, len(p) + 1):
            if p[j - 1] == '*':
                dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
            else:
                dp[i][j] = dp[i - 1][j - 1] and p[j - 1] in ('?', s[i - 1])
    return dp[len(s)][len(p)]'''),

("regex",
 "Write a Python function `regex_match(s: str, p: str) -> bool` that says whether p matches the whole of s, where '.' matches any one character and 'x*' matches zero or more of the preceding element. Do not use the re module.",
 r'''assert not regex_match('aa', 'a')
assert regex_match('aa', 'a*')
assert regex_match('ab', '.*')
assert regex_match('aab', 'c*a*b')
assert not regex_match('mississippi', 'mis*is*p*.')
assert regex_match('', 'a*b*')
assert not regex_match('ab', '.*c')''',
 r'''def regex_match(s, p):
    from functools import lru_cache
    @lru_cache(None)
    def m(i, j):
        if j == len(p):
            return i == len(s)
        first = i < len(s) and p[j] in (s[i], '.')
        if j + 1 < len(p) and p[j + 1] == '*':
            return m(i, j + 2) or (first and m(i + 1, j))
        return first and m(i + 1, j + 1)
    return m(0, 0)'''),

("razdalja",
 "Write a Python function `edit_distance(a: str, b: str) -> int` that returns the Levenshtein distance: the fewest single-character insertions, deletions and substitutions that turn a into b.",
 r'''assert edit_distance('horse', 'ros') == 3
assert edit_distance('intention', 'execution') == 5
assert edit_distance('', 'abc') == 3
assert edit_distance('kitten', 'sitting') == 3
assert edit_distance('same', 'same') == 0''',
 r'''def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]'''),

("topo",
 "Write a Python function `topo_order(n: int, edges: list[tuple[int, int]]) -> list[int] | None` for tasks 0..n-1, where an edge (a, b) means a must come before b. Return the lexicographically smallest valid order of all tasks, or None if there is a cycle.",
 r'''assert topo_order(4, [(1,0),(2,0),(3,1),(3,2)]) == [3,1,2,0]
assert topo_order(3, []) == [0,1,2]
assert topo_order(2, [(0,1),(1,0)]) is None
assert topo_order(5, [(4,0),(3,0),(2,1),(0,1)]) == [2,3,4,0,1]
assert topo_order(1, [(0,0)]) is None''',
 r'''def topo_order(n, edges):
    import heapq
    indeg = [0] * n
    out = [[] for _ in range(n)]
    for a, b in edges:
        out[a].append(b)
        indeg[b] += 1
    heap = [i for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        x = heapq.heappop(heap)
        order.append(x)
        for y in out[x]:
            indeg[y] -= 1
            if indeg[y] == 0:
                heapq.heappush(heap, y)
    return order if len(order) == n else None'''),

("mediana",
 "Write a Python function `median_sliding_window(nums: list[int], k: int) -> list[float]` that returns the median of every window of k consecutive numbers, as floats (for even k the mean of the two middle values).",
 r'''assert median_sliding_window([1,3,-1,-3,5,3,6,7], 3) == [1.0,-1.0,-1.0,3.0,5.0,6.0]
assert median_sliding_window([1,2,3,4,2,3,1,4,2], 3) == [2.0,3.0,3.0,3.0,2.0,3.0,2.0]
assert median_sliding_window([1,4,2,3], 4) == [2.5]
assert median_sliding_window([5,2], 2) == [3.5]
assert median_sliding_window([4,1,7], 1) == [4.0,1.0,7.0]
import random
random.seed(5)
for _ in range(30):
    xs = [random.randint(-50, 50) for _ in range(random.randint(1, 40))]
    k = random.randint(1, len(xs))
    want = []
    for i in range(len(xs) - k + 1):
        w = sorted(xs[i:i + k])
        want.append(float(w[k // 2]) if k % 2 else (w[k // 2 - 1] + w[k // 2]) / 2)
    assert median_sliding_window(xs, k) == want''',
 r'''def median_sliding_window(nums, k):
    import bisect
    w = sorted(nums[:k])
    res = []
    for i in range(k, len(nums) + 1):
        res.append(float(w[k // 2]) if k % 2 else (w[k // 2 - 1] + w[k // 2]) / 2)
        if i == len(nums):
            break
        w.pop(bisect.bisect_left(w, nums[i - k]))
        bisect.insort(w, nums[i])
    return res'''),

("rle",
 "Write two Python functions: `compress(s: str) -> str` that run-length encodes a string of letters, writing a letter followed by its run length only when the run is longer than 1 ('aaabccdddd' -> 'a3bc2d4'), and `decompress(s: str) -> str` that reverses it (run lengths may have several digits).",
 r'''assert compress('aaabccdddd') == 'a3bc2d4'
assert compress('') == ''
assert compress('abc') == 'abc'
assert compress('a' * 12 + 'b') == 'a12b'
assert decompress('a3bc2d4') == 'aaabccdddd'
assert decompress('x10y') == 'x' * 10 + 'y'
assert decompress('') == ''
import random
random.seed(3)
for _ in range(50):
    s = ''.join(random.choice('ab') * random.randint(1, 13) for _ in range(random.randint(0, 6)))
    assert decompress(compress(s)) == s''',
 r'''def compress(s):
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(s[i] + (str(j - i) if j - i > 1 else ''))
        i = j
    return ''.join(out)


def decompress(s):
    out, i = [], 0
    while i < len(s):
        ch = s[i]
        i += 1
        j = i
        while j < len(s) and s[j].isdigit():
            j += 1
        out.append(ch * (int(s[i:j]) if j > i else 1))
        i = j
    return ''.join(out)'''),

("verzije",
 "Write a Python function `compare_versions(a: str, b: str) -> int` that compares dotted version strings: -1 if a < b, 1 if a > b, 0 if equal. Revisions are compared as integers (leading zeros do not matter) and missing revisions count as 0, so '1.0' equals '1.0.0'.",
 r'''assert compare_versions('1.01', '1.001') == 0
assert compare_versions('1.0', '1.0.0') == 0
assert compare_versions('0.1', '1.1') == -1
assert compare_versions('1.10', '1.9') == 1
assert compare_versions('1.0.1', '1') == 1
assert compare_versions('7.5.2.4', '7.5.3') == -1''',
 r'''def compare_versions(a, b):
    x = [int(p) for p in a.split('.')]
    y = [int(p) for p in b.split('.')]
    n = max(len(x), len(y))
    x += [0] * (n - len(x))
    y += [0] * (n - len(y))
    return (x > y) - (x < y)'''),

("anagrami",
 "Write a Python function `group_anagrams(words: list[str]) -> list[list[str]]` that groups words that are anagrams of each other. Sort each group alphabetically and the groups by their first word; keep duplicate words.",
 r'''assert group_anagrams(['eat','tea','tan','ate','nat','bat']) == [['ate','eat','tea'],['bat'],['nat','tan']]
assert group_anagrams([]) == []
assert group_anagrams(['a','a']) == [['a','a']]
assert group_anagrams(['abc','cba','bca','xyz']) == [['abc','bca','cba'],['xyz']]''',
 r'''def group_anagrams(words):
    groups = {}
    for w in words:
        groups.setdefault(''.join(sorted(w)), []).append(w)
    return sorted(sorted(g) for g in groups.values())'''),

("ipv4",
 "Write a Python function `valid_ipv4(s: str) -> bool`: exactly four decimal parts separated by dots, each 0-255, digits only, and no leading zeros (except the single digit 0).",
 r'''assert valid_ipv4('192.168.1.1')
assert valid_ipv4('0.0.0.0')
assert valid_ipv4('255.255.255.255')
for bad in ['256.1.1.1', '01.1.1.1', '1.1.1', '1.1.1.1.', '1.2.3.-4', '1.2.3.4a', '1. 2.3.4', '', '1..2.3', '1.2.3.4.5']:
    assert not valid_ipv4(bad), bad''',
 r'''def valid_ipv4(s):
    parts = s.split('.')
    if len(parts) != 4:
        return False
    for p in parts:
        if not p or not all(c in '0123456789' for c in p) or (len(p) > 1 and p[0] == '0') or int(p) > 255:
            return False
    return True'''),

("rotacija",
 "Write a Python function `rotate_clockwise(m: list[list[int]]) -> list[list[int]]` that returns a new matrix rotated 90 degrees clockwise. The matrix may be rectangular (r rows and c columns become c rows and r columns). Do not modify the input.",
 r'''a = [[1,2,3],[4,5,6]]
assert rotate_clockwise(a) == [[4,1],[5,2],[6,3]] and a == [[1,2,3],[4,5,6]]
assert rotate_clockwise([[1]]) == [[1]]
assert rotate_clockwise([]) == []
assert rotate_clockwise([[1,2],[3,4]]) == [[3,1],[4,2]]''',
 r'''def rotate_clockwise(m):
    if not m:
        return []
    return [[m[r][c] for r in range(len(m) - 1, -1, -1)] for c in range(len(m[0]))]'''),

("nahrbtnik",
 "Write a Python function `knapsack(weights: list[int], values: list[int], capacity: int) -> int` that returns the largest total value of items whose total weight is at most capacity, each item used at most once.",
 r'''assert knapsack([1,3,4,5], [1,4,5,7], 7) == 9
assert knapsack([10,20,30], [60,100,120], 50) == 220
assert knapsack([], [], 10) == 0
assert knapsack([5], [10], 4) == 0''',
 r'''def knapsack(weights, values, capacity):
    best = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        for c in range(capacity, w - 1, -1):
            best[c] = max(best[c], best[c - w] + v)
    return best[capacity]'''),

("kovanci",
 "Write a Python function `coin_change(coins: list[int], amount: int) -> int` that returns the fewest coins that add up to amount (each coin value can be used any number of times), or -1 if it cannot be done.",
 r'''assert coin_change([1,2,5], 11) == 3
assert coin_change([2], 3) == -1
assert coin_change([1], 0) == 0
assert coin_change([186,419,83,408], 6249) == 20''',
 r'''def coin_change(coins, amount):
    INF = float('inf')
    best = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and best[a - c] + 1 < best[a]:
                best[a] = best[a - c] + 1
    return best[amount] if best[amount] != INF else -1'''),

("csv",
 "Write a Python function `parse_csv_line(line: str) -> list[str]` that splits one CSV line into fields: fields are separated by commas, a field may be enclosed in double quotes (then it may contain commas, and a doubled quote \"\" inside it stands for one quote), spaces are kept as they are, and an empty line is one empty field. Do not use the csv module.",
 r'''assert parse_csv_line('a,b,c') == ['a','b','c']
assert parse_csv_line('"a,b",c') == ['a,b','c']
assert parse_csv_line('"he said ""hi""",x') == ['he said "hi"','x']
assert parse_csv_line('') == ['']
assert parse_csv_line('a,,b') == ['a','','b']
assert parse_csv_line('a,') == ['a','']
assert parse_csv_line('"a""b"') == ['a"b']
assert parse_csv_line(' a , b ') == [' a ',' b ']''',
 r'''def parse_csv_line(line):
    fields, cur, i, quoted = [], [], 0, False
    while i < len(line):
        ch = line[i]
        if quoted:
            if ch == '"':
                if i + 1 < len(line) and line[i + 1] == '"':
                    cur.append('"')
                    i += 1
                else:
                    quoted = False
            else:
                cur.append(ch)
        elif ch == '"':
            quoted = True
        elif ch == ',':
            fields.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    fields.append(''.join(cur))
    return fields'''),

("ploscati",
 "Write a Python function `flatten(d: dict, sep: str = '.') -> dict` that flattens nested dicts: {'a': {'b': 1, 'c': {'d': 2}}, 'e': 3} -> {'a.b': 1, 'a.c.d': 2, 'e': 3}. Lists and other values are kept as they are (dicts inside lists are not flattened), and an empty nested dict stays as the value of its key.",
 r'''assert flatten({'a': {'b': 1, 'c': {'d': 2}}, 'e': 3}) == {'a.b': 1, 'a.c.d': 2, 'e': 3}
assert flatten({'a': {}}) == {'a': {}}
assert flatten({'x': {'y': [1, {'z': 2}]}}) == {'x.y': [1, {'z': 2}]}
assert flatten({'a': {'b': 1}}, sep='/') == {'a/b': 1}
assert flatten({}) == {}''',
 r'''def flatten(d, sep='.'):
    out = {}
    def walk(prefix, v):
        if isinstance(v, dict) and v:
            for k, x in v.items():
                walk(prefix + sep + k if prefix else k, x)
        else:
            out[prefix] = v
    for k, v in d.items():
        walk(k, v)
    return out'''),

("drseci_max",
 "Write a Python function `sliding_max(nums: list[int], k: int) -> list[int]` that returns the maximum of every window of k consecutive numbers, in O(n).",
 r'''assert sliding_max([1,3,-1,-3,5,3,6,7], 3) == [3,3,5,5,6,7]
assert sliding_max([1], 1) == [1]
assert sliding_max([9,8,7,6], 2) == [9,8,7]
import random
random.seed(7)
for _ in range(50):
    xs = [random.randint(-9, 9) for _ in range(random.randint(1, 30))]
    k = random.randint(1, len(xs))
    assert sliding_max(xs, k) == [max(xs[i:i + k]) for i in range(len(xs) - k + 1)]''',
 r'''def sliding_max(nums, k):
    from collections import deque
    q, res = deque(), []
    for i, x in enumerate(nums):
        while q and nums[q[-1]] <= x:
            q.pop()
        q.append(i)
        if q[0] <= i - k:
            q.popleft()
        if i >= k - 1:
            res.append(nums[q[0]])
    return res'''),

("voda",
 "Write a Python function `trap_water(heights: list[int]) -> int` that returns how much rain water is trapped between bars of the given heights (each bar is 1 wide).",
 r'''assert trap_water([0,1,0,2,1,0,1,3,2,1,2,1]) == 6
assert trap_water([4,2,0,3,2,5]) == 9
assert trap_water([]) == 0
assert trap_water([3]) == 0
assert trap_water([5,4,1,2]) == 1''',
 r'''def trap_water(heights):
    l, r = 0, len(heights) - 1
    lmax = rmax = water = 0
    while l < r:
        if heights[l] < heights[r]:
            lmax = max(lmax, heights[l])
            water += lmax - heights[l]
            l += 1
        else:
            rmax = max(rmax, heights[r])
            water += rmax - heights[r]
            r -= 1
    return water'''),

("pravokotnik",
 "Write a Python function `largest_rectangle(heights: list[int]) -> int` that returns the area of the largest rectangle in a histogram whose bars have the given heights and width 1.",
 r'''assert largest_rectangle([2,1,5,6,2,3]) == 10
assert largest_rectangle([2,4]) == 4
assert largest_rectangle([]) == 0
assert largest_rectangle([1,1,1,1]) == 4
assert largest_rectangle([6,2,5,4,5,1,6]) == 12''',
 r'''def largest_rectangle(heights):
    stack, best = [], 0
    for i, h in enumerate(heights + [0]):
        start = i
        while stack and stack[-1][1] >= h:
            j, hh = stack.pop()
            best = max(best, hh * (i - j))
            start = j
        stack.append((start, h))
    return best'''),

("dekodiranja",
 "Write a Python function `num_decodings(s: str) -> int` that counts the ways to decode a string of digits where 'A' = '1', ..., 'Z' = '26' ('226' can be 'BZ', 'VF' or 'BBF', so 3). A '0' cannot start a code, so '06' has 0 ways.",
 r'''assert num_decodings('226') == 3
assert num_decodings('06') == 0
assert num_decodings('11106') == 2
assert num_decodings('0') == 0
assert num_decodings('10') == 1
assert num_decodings('2101') == 1
assert num_decodings('27') == 1''',
 r'''def num_decodings(s):
    if not s:
        return 0
    prev, cur = 1, (1 if s[0] != '0' else 0)
    for i in range(1, len(s)):
        nxt = cur if s[i] != '0' else 0
        if s[i - 1] == '1' or (s[i - 1] == '2' and s[i] <= '6'):
            nxt += prev
        prev, cur = cur, nxt
    return cur'''),

("tabela",
 "Write a Python function `format_table(rows: list[list]) -> str`. The first row is the header. Each column is as wide as its widest cell (the str() of the value). Cells are separated by ' | '. Cells holding int or float values (but not bool) are right-aligned; everything else, and every header cell, is left-aligned. After the header comes a line of '-' as wide as each column, with '-+-' between columns. Join the lines with a newline, with no newline at the end, and do not strip trailing spaces.",
 r'''rows = [['name', 'qty', 'price'], ['apple', 3, 1.5], ['kiwi', 12, 0.25]]
assert format_table(rows) == 'name  | qty | price\n------+-----+------\napple |   3 |   1.5\nkiwi  |  12 |  0.25'
assert format_table([['a', 'b'], ['x', True]]) == 'a | b   \n--+-----\nx | True'
assert format_table([['h']]) == 'h\n-' ''',
 r'''def format_table(rows):
    cells = [[str(v) for v in row] for row in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(cells[0]))]
    def line(row, raw, header):
        out = []
        for i, (c, v) in enumerate(zip(row, raw)):
            num = not header and isinstance(v, (int, float)) and not isinstance(v, bool)
            out.append(c.rjust(widths[i]) if num else c.ljust(widths[i]))
        return ' | '.join(out)
    lines = [line(cells[0], rows[0], True), '-+-'.join('-' * w for w in widths)]
    lines += [line(c, r, False) for c, r in zip(cells[1:], rows[1:])]
    return '\n'.join(lines)'''),

("pot_json",
 "Write a Python function `get_path(obj, path: str)` that follows a path like 'a.b[2].c' through nested dicts and lists: names separated by '.' are dict keys, '[n]' (n a non-negative integer) is a list index, and several indexes may follow each other ('m[0][1]'). Raise KeyError for a missing key, an index out of range, or a step of the wrong kind (a key on a list or an index on a dict).",
 r'''d = {'a': {'b': [10, 20, {'c': 'x'}]}, 'm': [[1, 2], [3, 4]]}
assert get_path(d, 'a.b[2].c') == 'x'
assert get_path(d, 'a.b[0]') == 10
assert get_path(d, 'm[1][0]') == 3
assert get_path(d, 'a') == {'b': [10, 20, {'c': 'x'}]}
for bad in ['a.x', 'a.b[5]', 'm.b', 'a[0]']:
    try:
        get_path(d, bad)
        raise SystemExit('no KeyError for ' + bad)
    except KeyError:
        pass''',
 r'''def get_path(obj, path):
    import re
    cur = obj
    for part in path.split('.'):
        m = re.fullmatch(r'([^\[\]]*)((?:\[\d+\])*)', part)
        if not m:
            raise KeyError(part)
        name, idx = m.group(1), m.group(2)
        if name:
            if not isinstance(cur, dict) or name not in cur:
                raise KeyError(name)
            cur = cur[name]
        for n in re.findall(r'\[(\d+)\]', idx):
            n = int(n)
            if not isinstance(cur, list) or n >= len(cur):
                raise KeyError(n)
            cur = cur[n]
    return cur'''),

("slug",
 "Napiši funkcijo v Pythonu `slugify(besedilo: str) -> str`, ki iz besedila naredi oznako za URL: vse z malimi črkami, č→c, š→s, ž→z, ć→c, đ→d (tudi velike črke), vsako zaporedje znakov, ki niso črke a-z ali števke 0-9, postane en sam '-', na začetku in koncu pa ni '-'.",
 r'''assert slugify('Škofja Loka – Žiri!') == 'skofja-loka-ziri'
assert slugify('  Čaj   in   kava ') == 'caj-in-kava'
assert slugify('Đurđevac 2024') == 'durdevac-2024'
assert slugify('') == ''
assert slugify('---') == ''
assert slugify('Ćevapčići & co.') == 'cevapcici-co'
assert slugify('Ana_Marija') == 'ana-marija' ''',
 r'''def slugify(besedilo):
    import re
    t = besedilo.lower()
    for a, b in (('č', 'c'), ('š', 's'), ('ž', 'z'), ('ć', 'c'), ('đ', 'd')):
        t = t.replace(a, b)
    return re.sub(r'[^a-z0-9]+', '-', t).strip('-')'''),

("davcna",
 "Napiši funkcijo v Pythonu `davcna_veljavna(s: str) -> bool` za slovensko davčno številko: natanko 8 števk, prva ni 0. Prvih 7 števk pomnoži z utežmi 8, 7, 6, 5, 4, 3, 2 in jih seštej. Kontrolna števka je 11 − (vsota mod 11); če to da 10, je kontrolna števka 0, če da 11, številka ni veljavna. Številka je veljavna, ko je zadnja števka enaka kontrolni. Presledki ali drugi znaki pomenijo neveljavno številko.",
 r'''assert davcna_veljavna('10000003')
assert davcna_veljavna('12345679')
assert not davcna_veljavna('12345678')
assert davcna_veljavna('10800000')
assert not davcna_veljavna('10600000')
assert not davcna_veljavna('10600001')
assert not davcna_veljavna('01234567')
assert not davcna_veljavna('1234567')
assert not davcna_veljavna('1234567a')
assert not davcna_veljavna('12345679 ')''',
 r'''def davcna_veljavna(s):
    if len(s) != 8 or not all(c in '0123456789' for c in s) or s[0] == '0':
        return False
    vsota = sum(int(c) * w for c, w in zip(s[:7], (8, 7, 6, 5, 4, 3, 2)))
    k = 11 - vsota % 11
    if k == 11:
        return False
    if k == 10:
        k = 0
    return int(s[7]) == k'''),

("iban",
 "Write a Python function `iban_valid(s: str) -> bool`: ignore spaces and letter case; after that the IBAN must be 15 to 34 letters and digits, starting with two letters and two digits. Move the first four characters to the end, replace each letter with two digits (A=10, B=11, ..., Z=35), and the resulting number mod 97 must be 1.",
 r'''assert iban_valid('SI56 1910 0000 0123 438')
assert iban_valid('GB82 WEST 1234 5698 7654 32')
assert iban_valid('gb82west12345698765432')
assert not iban_valid('GB82 WEST 1234 5698 7654 33')
assert not iban_valid('SI56 1910 0000 0123 439')
assert not iban_valid('GB82-WEST-1234-5698-7654-32')
assert not iban_valid('XX')
assert not iban_valid('')''',
 r'''def iban_valid(s):
    t = s.replace(' ', '').upper()
    if not (15 <= len(t) <= 34) or not t.isascii() or not t.isalnum():
        return False
    if not (t[:2].isalpha() and t[2:4].isdigit()):
        return False
    r = t[4:] + t[:4]
    num = ''.join(str(ord(c) - 55) if c.isalpha() else c for c in r)
    return int(num) % 97 == 1'''),

("lfu",
 "Write a Python class `LFUCache` with `__init__(self, capacity: int)`, `get(self, key: int) -> int` (-1 if missing) and `put(self, key: int, value: int) -> None`. When the cache is full, put evicts the least frequently used key, and among keys with the same use count the least recently used one. A get and a put (also a put that updates an existing key) count as one use. A capacity of 0 stores nothing.",
 r'''c = LFUCache(2)
c.put(1, 1); c.put(2, 2)
assert c.get(1) == 1
c.put(3, 3)
assert c.get(2) == -1
assert c.get(3) == 3
c.put(4, 4)
assert c.get(1) == -1
assert c.get(3) == 3
assert c.get(4) == 4
z = LFUCache(0)
z.put(0, 0)
assert z.get(0) == -1
u = LFUCache(2)
u.put(2, 1); u.put(2, 2)
assert u.get(2) == 2
u.put(1, 1); u.put(4, 1)
assert u.get(2) == 2 and u.get(1) == -1 and u.get(4) == 1''',
 r'''class LFUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.val, self.cnt, self.tick, self.t = {}, {}, {}, 0
    def _use(self, key):
        self.t += 1
        self.cnt[key] = self.cnt.get(key, 0) + 1
        self.tick[key] = self.t
    def get(self, key):
        if key not in self.val:
            return -1
        self._use(key)
        return self.val[key]
    def put(self, key, value):
        if self.cap <= 0:
            return
        if key not in self.val and len(self.val) >= self.cap:
            victim = min(self.val, key=lambda k: (self.cnt[k], self.tick[k]))
            for d in (self.val, self.cnt, self.tick):
                del d[victim]
        self.val[key] = value
        self._use(key)'''),

("sobe",
 "Write a Python function `min_meeting_rooms(intervals: list[list[int]]) -> int` that returns how many rooms are needed so that no two meetings [start, end) in the same room overlap. A meeting that ends at time t frees its room for one that starts at t.",
 r'''assert min_meeting_rooms([[0,30],[5,10],[15,20]]) == 2
assert min_meeting_rooms([[7,10],[2,4]]) == 1
assert min_meeting_rooms([]) == 0
assert min_meeting_rooms([[1,5],[5,10]]) == 1
assert min_meeting_rooms([[1,10],[2,7],[3,19],[8,12],[10,20],[11,30]]) == 4''',
 r'''def min_meeting_rooms(intervals):
    ev = sorted([(s, 1) for s, e in intervals] + [(e, -1) for s, e in intervals])
    cur = best = 0
    for _, d in ev:
        cur += d
        best = max(best, cur)
    return best'''),

("nivoji",
 "Write a Python function `level_order(values: list) -> list[list[int]]` where values describes a binary tree in level order the way LeetCode does (for example [3,9,20,None,None,15,7]; the children of a missing node are not listed). Return the node values level by level, left to right.",
 r'''assert level_order([3,9,20,None,None,15,7]) == [[3],[9,20],[15,7]]
assert level_order([]) == []
assert level_order([1]) == [[1]]
assert level_order([1,2,3,4,None,None,5]) == [[1],[2,3],[4,5]]
assert level_order([1,None,2,None,3]) == [[1],[2],[3]]
assert level_order([5,4,8,11,None,13,4,7,2,None,None,5,1]) == [[5],[4,8],[11,13,4],[7,2,5,1]]''',
 r'''def level_order(values):
    if not values or values[0] is None:
        return []
    levels, cur, i = [], [values[0]], 1
    while cur:
        levels.append(cur)
        nxt = []
        for _ in cur:
            for _ in range(2):
                if i < len(values):
                    if values[i] is not None:
                        nxt.append(values[i])
                    i += 1
        cur = nxt
    return levels'''),

("rpn",
 "Write a Python function `to_rpn(expr: str) -> str` that converts an infix expression to Reverse Polish Notation. Numbers are non-negative integers (possibly several digits), operators are + - * / ^ (^ binds tightest and is right-associative, the others are left-associative with * and / above + and -), there may be parentheses and spaces. Return the tokens separated by single spaces.",
 r'''assert to_rpn('3 + 4 * 2 / ( 1 - 5 ) ^ 2 ^ 3') == '3 4 2 * 1 5 - 2 3 ^ ^ / +'
assert to_rpn('1+2+3') == '1 2 + 3 +'
assert to_rpn('2^3^2') == '2 3 2 ^ ^'
assert to_rpn('(10+20)*30') == '10 20 + 30 *'
assert to_rpn('42') == '42'
assert to_rpn('8/4/2') == '8 4 / 2 /' ''',
 r'''def to_rpn(expr):
    import re
    prec = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    out, ops = [], []
    for tok in re.findall(r'\d+|[-+*/^()]', expr):
        if tok.isdigit():
            out.append(tok)
        elif tok == '(':
            ops.append(tok)
        elif tok == ')':
            while ops[-1] != '(':
                out.append(ops.pop())
            ops.pop()
        else:
            while ops and ops[-1] != '(' and (prec[ops[-1]] > prec[tok] or (prec[ops[-1]] == prec[tok] and tok != '^')):
                out.append(ops.pop())
            ops.append(tok)
    while ops:
        out.append(ops.pop())
    return ' '.join(out)'''),

("cikcak",
 "Write a Python function `zigzag(s: str, rows: int) -> str`: write s in a zigzag over the given number of rows (down the rows, then diagonally up, and so on) and read it row by row. zigzag('PAYPALISHIRING', 3) == 'PAHNAPLSIIGYIR'.",
 r'''assert zigzag('PAYPALISHIRING', 3) == 'PAHNAPLSIIGYIR'
assert zigzag('PAYPALISHIRING', 4) == 'PINALSIGYAHRPI'
assert zigzag('AB', 1) == 'AB'
assert zigzag('A', 5) == 'A'
assert zigzag('', 2) == '' ''',
 r'''def zigzag(s, rows):
    if rows == 1 or rows >= len(s):
        return s
    lines = [''] * rows
    r, step = 0, 1
    for ch in s:
        lines[r] += ch
        if r == 0:
            step = 1
        elif r == rows - 1:
            step = -1
        r += step
    return ''.join(lines)'''),

("poravnava",
 "Write a Python function `justify(words: list[str], width: int) -> list[str]` that formats words into lines of exactly width characters, fully justified: pack as many words as fit (one space between them at least) greedily; spread the extra spaces of a line as evenly as possible between its words, with more on the left gaps when they do not divide evenly. The last line, and any line with a single word, is left-justified with single spaces and padded with spaces on the right. Every word is at most width long.",
 r'''assert justify(['This','is','an','example','of','text','justification.'], 16) == ['This    is    an','example  of text','justification.  ']
assert justify(['What','must','be','acknowledgment','shall','be'], 16) == ['What   must   be','acknowledgment  ','shall be        ']
assert justify(['Science','is','what','we','understand','well','enough','to','explain','to','a','computer.','Art','is','everything','else','we','do'], 20) == ['Science  is  what we','understand      well','enough to explain to','a  computer.  Art is','everything  else  we','do                  ']''',
 r'''def justify(words, width):
    lines, i = [], 0
    while i < len(words):
        j, length = i, 0
        while j < len(words) and length + len(words[j]) + (j - i) <= width:
            length += len(words[j])
            j += 1
        line = words[i:j]
        if j == len(words) or len(line) == 1:
            s = ' '.join(line)
            lines.append(s + ' ' * (width - len(s)))
        else:
            gaps = len(line) - 1
            spaces = width - length
            s = ''
            for k, w in enumerate(line[:-1]):
                s += w + ' ' * (spaces // gaps + (1 if k < spaces % gaps else 0))
            lines.append(s + line[-1])
        i = j
    return lines'''),

("diff",
 "Write a Python function `diff(a: list[str], b: list[str]) -> list[str]` that returns a shortest line diff from a to b: each output line is ' ' + line for a line kept from both, '-' + line for a line only in a, and '+' + line for a line only in b, in order, so that the kept lines form a longest common subsequence of a and b.",
 r'''def _lcs(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) - 1, -1, -1):
        for j in range(len(b) - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    return dp[0][0]
def _check(a, b):
    d = diff(a, b)
    assert all(isinstance(x, str) and x[:1] in (' ', '-', '+') for x in d), d
    assert [x[1:] for x in d if x[0] in ' -'] == a, d
    assert [x[1:] for x in d if x[0] in ' +'] == b, d
    assert sum(1 for x in d if x[0] == ' ') == _lcs(a, b), d
_check(['a','b','c','d'], ['a','c','d','e'])
_check([], ['x'])
_check(['x','y'], [])
_check(list('ABCBDAB'), list('BDCABA'))
_check([' x', 'y'], [' x', 'z'])
import random
random.seed(11)
for _ in range(40):
    a = [random.choice('pqrs') for _ in range(random.randint(0, 9))]
    b = [random.choice('pqrs') for _ in range(random.randint(0, 9))]
    _check(a, b)''',
 r'''def diff(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) - 1, -1, -1):
        for j in range(len(b) - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(' ' + a[i]); i += 1; j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            out.append('-' + a[i]); i += 1
        else:
            out.append('+' + b[j]); j += 1
    out += ['-' + x for x in a[i:]] + ['+' + x for x in b[j:]]
    return out'''),

("poizvedba",
 "Write a Python function `parse_query(qs: str) -> dict[str, list[str]]` for a URL query string such as 'a=1&b=x+y&a=%C4%8D': split on '&' and skip empty parts, split each part on its first '=' (a part without '=' has the value ''), decode '+' as a space and %XX escapes as UTF-8 bytes in names and values, and collect repeated names' values in order. Do not use urllib.",
 r'''assert parse_query('a=1&b=x+y&a=%C4%8D') == {'a': ['1', 'č'], 'b': ['x y']}
assert parse_query('') == {}
assert parse_query('flag&&x=') == {'flag': [''], 'x': ['']}
assert parse_query('k=a%3Db%26c') == {'k': ['a=b&c']}
assert parse_query('%C5%BEaba=1') == {'žaba': ['1']}
assert parse_query('a=1=2') == {'a': ['1=2']}''',
 r'''def parse_query(qs):
    def dec(s):
        s = s.replace('+', ' ')
        out, i = bytearray(), 0
        while i < len(s):
            if s[i] == '%' and i + 2 < len(s) + 0 and len(s[i + 1:i + 3]) == 2:
                out += bytes([int(s[i + 1:i + 3], 16)])
                i += 3
            else:
                out += s[i].encode()
                i += 1
        return out.decode('utf-8')
    res = {}
    for part in qs.split('&'):
        if not part:
            continue
        k, _, v = part.partition('=')
        res.setdefault(dec(k), []).append(dec(v))
    return res'''),

("predloga",
 "Write a Python function `render(tpl: str, ctx: dict) -> str` that replaces every '{{ name }}' (spaces inside the braces are optional) with the str() of its value in ctx; a dotted name such as 'a.b' looks up nested dicts; a missing name raises KeyError.",
 r'''assert render('Hi {{name}}!', {'name': 'Ana'}) == 'Hi Ana!'
assert render('{{ a.b }}-{{a.c}}', {'a': {'b': 1, 'c': 2}}) == '1-2'
assert render('no tags', {}) == 'no tags'
assert render('{{x}}{{ x }}', {'x': 0}) == '00'
try:
    render('{{ y }}', {'x': 1})
    raise SystemExit('no KeyError')
except KeyError:
    pass
try:
    render('{{ a.z }}', {'a': {'b': 1}})
    raise SystemExit('no KeyError')
except KeyError:
    pass''',
 r'''def render(tpl, ctx):
    import re
    def look(m):
        cur = ctx
        for part in m.group(1).split('.'):
            if not isinstance(cur, dict) or part not in cur:
                raise KeyError(m.group(1))
            cur = cur[part]
        return str(cur)
    return re.sub(r'\{\{\s*([\w.]+)\s*\}\}', look, tpl)'''),

("razdeli",
 "Napiši funkcijo v Pythonu `razdeli(znesek_centov: int, delezi: list[int]) -> list[int]`, ki znesek v centih razdeli sorazmerno z deleži (vsota deležev je pozitivna). Vsak najprej dobi navzdol zaokrožen del, preostale cente pa po en cent dobijo tisti z največjim ostankom pri deljenju; pri enakem ostanku ima prednost tisti, ki je prej v seznamu. Vsota rezultata je vedno enaka znesku. Računaj s celimi števili.",
 r'''assert razdeli(100, [1, 1, 1]) == [34, 33, 33]
assert razdeli(1000, [1, 2, 3]) == [167, 333, 500]
assert razdeli(5, [1, 1]) == [3, 2]
assert razdeli(0, [1, 2]) == [0, 0]
assert razdeli(7, [0, 1]) == [0, 7]
assert razdeli(10, [3, 3, 3, 1]) == [3, 3, 3, 1]
assert razdeli(11, [1, 1, 1, 1]) == [3, 3, 3, 2]
assert sum(razdeli(99999, [7, 13, 29, 51])) == 99999''',
 r'''def razdeli(znesek_centov, delezi):
    skupaj = sum(delezi)
    deli = [znesek_centov * d // skupaj for d in delezi]
    ostanki = [znesek_centov * d % skupaj for d in delezi]
    manjka = znesek_centov - sum(deli)
    vrstni = sorted(range(len(delezi)), key=lambda i: (-ostanki[i], i))
    for i in vrstni[:manjka]:
        deli[i] += 1
    return deli'''),

("delovni",
 "Napiši funkcijo v Pythonu `delovni_dnevi(od: datetime.date, do: datetime.date, prazniki: set) -> int`, ki prešteje delovne dni (od ponedeljka do petka, ki niso v množici praznikov) od `od` do `do`, oba dneva vključno. Če je `od` po `do`, vrni 0. Datumi so lahko več let narazen.",
 r'''import datetime
d = datetime.date
assert delovni_dnevi(d(2026,11,16), d(2026,11,22), set()) == 5
assert delovni_dnevi(d(2026,11,16), d(2026,11,22), {d(2026,11,18)}) == 4
assert delovni_dnevi(d(2026,11,21), d(2026,11,21), set()) == 0
assert delovni_dnevi(d(2026,11,22), d(2026,11,16), set()) == 0
assert delovni_dnevi(d(2026,12,24), d(2027,1,2), {d(2026,12,25), d(2026,12,26), d(2027,1,1), d(2027,1,2)}) == 5
assert delovni_dnevi(d(2024,1,1), d(2024,12,31), set()) == 262''',
 r'''def delovni_dnevi(od, do, prazniki):
    import datetime
    n, dan = 0, od
    while dan <= do:
        if dan.weekday() < 5 and dan not in prazniki:
            n += 1
        dan += datetime.timedelta(days=1)
    return n'''),
]


# ---------------------------------------------------------------- edits ------------------------------------------------

ZALOGA = '''"""Preprosta evidenca zaloge v skladiscu."""


class Zaloga:
    def __init__(self):
        self._artikli = {}

    def dodaj(self, ime, kolicina):
        if kolicina <= 0:
            raise ValueError("kolicina mora biti pozitivna")
        self._artikli[ime] = self._artikli.get(ime, 0) + kolicina

    def kolicina(self, ime):
        return self._artikli.get(ime, 0)

    def vsi(self):
        return sorted(self._artikli.items())


def iz_seznama(pari):
    z = Zaloga()
    for ime, kolicina in pari:
        z.dodaj(ime, kolicina)
    return z
'''

BESEDILO = '''"""Orodja za preprosto obdelavo besedila."""

import re

BESEDA = re.compile(r"[\\w']+")


def stej_besede(besedilo):
    return len(BESEDA.findall(besedilo))


def najdaljsa_beseda(besedilo):
    besede = BESEDA.findall(besedilo)
    return max(besede, key=len) if besede else ""


def povzetek(besedilo, najvec=5):
    besede = BESEDA.findall(besedilo)
    if stej_besede(besedilo) <= najvec:
        return besedilo.strip()
    return " ".join(besede[:najvec]) + " ..."


def statistika(odstavki):
    skupaj = sum(stej_besede(o) for o in odstavki)
    return {"odstavkov": len(odstavki), "besed": skupaj,
            "povprecje": skupaj / len(odstavki) if odstavki else 0.0}
'''

STATS = '''"""Small statistics helpers."""

from collections import Counter


def mean(xs):
    if not xs:
        raise ValueError("mean of empty list")
    return sum(xs) / len(xs)


def median(xs):
    if not xs:
        raise ValueError("median of empty list")
    s = sorted(xs)
    return s[len(s) // 2]


def mode(xs):
    if not xs:
        raise ValueError("mode of empty list")
    counts = Counter(xs)
    best = max(counts.values())
    return min(x for x, c in counts.items() if c == best)


def spread(xs):
    return max(xs) - min(xs)
'''

CENE = '''"""Izpis cen in racunov."""


def formatiraj_ceno(znesek):
    centi = round(znesek * 100)
    evri, ostanek = divmod(abs(centi), 100)
    znak = "-" if centi < 0 else ""
    tisoci = f"{evri:,}".replace(",", ".")
    return f"{znak}{tisoci},{ostanek:02d} €"


def formatiraj_racun(postavke):
    vrstice = []
    skupaj = 0
    for ime, cena in postavke:
        vrstice.append(f"{ime}: {formatiraj_ceno(cena)}")
        skupaj += cena
    vrstice.append(f"Skupaj: {formatiraj_ceno(skupaj)}")
    return "\\n".join(vrstice)
'''

STUDENTI = '''"""Seznam studentov in ocen."""

from dataclasses import dataclass


@dataclass
class Student:
    ime: str
    priimek: str
    ocena: int


def preberi(vrstice):
    studenti = []
    for vrstica in vrstice:
        ime, priimek, ocena = vrstica.split(";")
        studenti.append(Student(ime.strip(), priimek.strip(), int(ocena)))
    return studenti


def razvrsti(studenti):
    return sorted(studenti, key=lambda s: s.priimek)


def najboljsi(studenti, n=3):
    return razvrsti(studenti)[:n]
'''

STRANI = '''"""Razdelitev seznama na strani."""

import math


def stevilo_strani(seznam, velikost):
    if velikost < 1:
        raise ValueError("velikost strani mora biti vsaj 1")
    return max(1, math.ceil(len(seznam) / velikost))


def stran(seznam, st_strani, velikost):
    if velikost < 1:
        raise ValueError("velikost strani mora biti vsaj 1")
    zacetek = st_strani * velikost
    return seznam[zacetek:zacetek + velikost]


def opis_strani(seznam, st_strani, velikost):
    return f"stran {st_strani} od {stevilo_strani(seznam, velikost)}"
'''

DDV_MOD = '''"""Racunanje cen z DDV."""

DDV = 0.22
ZNIZAN_DDV = 0.095


def z_ddv(neto, znizan=False):
    stopnja = ZNIZAN_DDV if znizan else DDV
    return round(neto * (1 + stopnja), 2)


def brez_ddv(bruto, znizan=False):
    stopnja = ZNIZAN_DDV if znizan else DDV
    return round(bruto / (1 + stopnja), 2)


def znesek_ddv(neto, znizan=False):
    return round(z_ddv(neto, znizan) - neto, 2)


def opis():
    return f"Splosna stopnja {DDV * 100:.1f} %, znizana {ZNIZAN_DDV * 100:.1f} %"


def stopnje():
    return {"splosna": DDV, "znizana": ZNIZAN_DDV}
'''

INI = '''"""A tiny INI-style config reader."""


def parse(text):
    data, section = {}, None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            data.setdefault(section, {})
            continue
        if "=" not in line:
            raise ValueError(f"bad line: {raw!r}")
        key, value = line.split("=", 1)
        if section is None:
            raise ValueError("key outside a section")
        data[section][key.strip()] = value.strip()
    return data


def get(data, section, key, default=None):
    return data.get(section, {}).get(key, default)
'''

CAS = '''"""Pretvorbe casa."""


def preberi_cas(niz):
    ure, minute = niz.split(":")
    ure, minute = int(ure), int(minute)
    if not (0 <= ure < 24 and 0 <= minute < 60):
        raise ValueError(f"neveljaven cas: {niz}")
    return ure * 3600 + minute * 60


def zapisi_cas(sekunde):
    ure, ostanek = divmod(sekunde, 3600)
    minute, sekunde = divmod(ostanek, 60)
    return f"{ure:02d}:{minute:02d}:{sekunde:02d}"


def trajanje(od, do):
    return preberi_cas(do) - preberi_cas(od)
'''

KOSARICA = '''"""Nakupovalna kosarica."""


class Kosarica:
    def __init__(self):
        self.postavke = []

    def dodaj(self, ime, cena, kolicina=1):
        if cena < 0 or kolicina < 1:
            raise ValueError("napacna cena ali kolicina")
        self.postavke.append((ime, cena, kolicina))

    def skupaj(self):
        return round(sum(c * k for _, c, k in self.postavke), 2)

    def stevilo_kosov(self):
        return sum(k for _, _, k in self.postavke)
'''

IMENIK = '''"""Telefonski imenik."""


class Imenik:
    def __init__(self):
        self._vnosi = {}

    def dodaj(self, ime, stevilka):
        self._vnosi[ime] = stevilka

    def najdi(self, ime):
        return self._vnosi.get(ime)

    def imena(self):
        return sorted(self._vnosi)
'''


def _blok(*imena):
    by = {f[0]: f[3] for f in FUNKCIJE}
    return "\n\n\n".join(by[i] for i in imena) + "\n"


def _testi(*imena):
    by = {f[0]: f[2] for f in FUNKCIJE}
    return "\n".join(by[i] for i in imena)


ORODJA_IMENA = ("spiral", "decode", "rle", "verzije", "ipv4", "voda", "dekodiranja", "cikcak", "pot", "razdalja", "anagrami", "rotacija")
RACUNI_IMENA = ("nahrbtnik", "kovanci", "pravokotnik", "drseci_max", "razdeli", "davcna", "iban", "tabela", "sobe", "csv")
ORODJA = '"""Zbirka pomoznih funkcij."""\n\n\n' + _blok(*ORODJA_IMENA)
_RAZDELI_NAPAKA = ("vrstni = sorted(range(len(delezi)), key=lambda i: (-ostanki[i], i))",
                   "vrstni = sorted(range(len(delezi)), key=lambda i: (-ostanki[i], -i))")
RACUNI = ('"""Racunske in preverjalne funkcije."""\n\n\n' + _blok(*RACUNI_IMENA)).replace(*_RAZDELI_NAPAKA)

# (id, mode, path, original, instruction, tests, ref_edit as a list of (old, new))
UREJANJE = [
("zaloga", "orodje", "/projekt/zaloga.py", ZALOGA,
 "V datoteki /projekt/zaloga.py dodaj razredu Zaloga metodo `odvzemi(ime, kolicina)`, ki zalogo zmanjša. Če količina ni pozitivna ali artikla ni dovolj, naj sproži ValueError in zaloge ne spremeni. Artikel, ki pade na 0, naj izgine iz `vsi()`.",
 r'''z = iz_seznama([('jabolka', 5), ('hruske', 2)])
z.odvzemi('jabolka', 3)
assert z.kolicina('jabolka') == 2
z.odvzemi('hruske', 2)
assert z.vsi() == [('jabolka', 2)]
for args in [('jabolka', 3), ('kivi', 1), ('jabolka', 0)]:
    try:
        z.odvzemi(*args)
        raise SystemExit('ni ValueError za ' + repr(args))
    except ValueError:
        pass
assert z.kolicina('jabolka') == 2
z.dodaj('kivi', 1)
assert z.vsi() == [('jabolka', 2), ('kivi', 1)]''',
 [("    def vsi(self):\n", "    def odvzemi(self, ime, kolicina):\n        if kolicina <= 0 or self._artikli.get(ime, 0) < kolicina:\n            raise ValueError(\"ni dovolj\")\n        self._artikli[ime] -= kolicina\n        if self._artikli[ime] == 0:\n            del self._artikli[ime]\n\n    def vsi(self):\n")]),

("preimenuj", "orodje", "/projekt/besedilo.py", BESEDILO,
 "Preimenuj funkcijo `stej_besede` v `prestej_besede` v datoteki /projekt/besedilo.py in popravi vse njene klice.",
 r'''assert prestej_besede('Ena dva tri') == 3
assert 'stej_besede' not in globals()
assert povzetek('a b c d e f g', 3) == 'a b c ...'
assert povzetek(' kratko ', 3) == 'kratko'
assert statistika(['a b', 'c d e']) == {'odstavkov': 2, 'besed': 5, 'povprecje': 2.5}
assert najdaljsa_beseda("to je preprosto") == 'preprosto' ''',
 [("def stej_besede(", "def prestej_besede("), ("if stej_besede(", "if prestej_besede("), ("sum(stej_besede(", "sum(prestej_besede(")]),

("mediana_fix", "orodje", "/project/stats.py", STATS,
 "median() in /project/stats.py returns the wrong value for lists of even length: it should return the mean of the two middle values. Fix it.",
 r'''assert median([1, 2, 3, 4]) == 2.5
assert median([3, 1, 2]) == 2
assert median([5]) == 5
assert median([4, 1]) == 2.5
assert mean([1, 2, 3, 4]) == 2.5
assert mode([1, 2, 2, 3, 3]) == 2
assert spread([3, 9, 1]) == 8
try:
    median([])
    raise SystemExit('no ValueError')
except ValueError:
    pass''',
 [("    return s[len(s) // 2]\n", "    n = len(s)\n    if n % 2:\n        return s[n // 2]\n    return (s[n // 2 - 1] + s[n // 2]) / 2\n")]),

("valuta", "orodje", "/projekt/cene.py", CENE,
 "V /projekt/cene.py dodaj funkciji `formatiraj_ceno` neobvezen parameter `valuta` s privzeto vrednostjo '€' in ga uporabi namesto fiksnega znaka. Tudi `formatiraj_racun` naj sprejme neobvezen `valuta` (privzeto '€') in ga poda naprej.",
 r'''assert formatiraj_ceno(1234.5) == '1.234,50 €'
assert formatiraj_ceno(-3, valuta='USD') == '-3,00 USD'
assert formatiraj_ceno(0.07, 'CHF') == '0,07 CHF'
assert formatiraj_racun([('kava', 1.2), ('sok', 2)]) == 'kava: 1,20 €\nsok: 2,00 €\nSkupaj: 3,20 €'
assert formatiraj_racun([('a', 1)], valuta='$') == 'a: 1,00 $\nSkupaj: 1,00 $' ''',
 [("def formatiraj_ceno(znesek):", "def formatiraj_ceno(znesek, valuta=\"€\"):"),
  ("{ostanek:02d} €\"", "{ostanek:02d} {valuta}\""),
  ("def formatiraj_racun(postavke):", "def formatiraj_racun(postavke, valuta=\"€\"):"),
  ("{formatiraj_ceno(cena)}", "{formatiraj_ceno(cena, valuta)}"),
  ("{formatiraj_ceno(skupaj)}", "{formatiraj_ceno(skupaj, valuta)}")]),

("razvrsti", "orodje", "/projekt/studenti.py", STUDENTI,
 "Funkcija `razvrsti` v /projekt/studenti.py naj študente razvrsti padajoče po oceni, pri enaki oceni pa naraščajoče po priimku (zdaj razvršča samo po priimku).",
 r'''s = preberi(['Ana; Novak; 9', 'Bor; Kos; 10', 'Cene; Bizjak; 9', 'Dan; Zupan; 6'])
assert [x.priimek for x in razvrsti(s)] == ['Kos', 'Bizjak', 'Novak', 'Zupan']
assert [x.ime for x in najboljsi(s, 2)] == ['Bor', 'Cene']
assert s[0] == Student('Ana', 'Novak', 9)''',
 [("key=lambda s: s.priimek)", "key=lambda s: (-s.ocena, s.priimek))")]),

("strani", "orodje", "/projekt/strani.py", STRANI,
 "Strani se v /projekt/strani.py štejejo od 1, a `stran(x, 1, 10)` vrne drugo stran. Popravi to in poskrbi, da `st_strani` manjši od 1 sproži ValueError.",
 r'''x = list(range(25))
assert stran(x, 1, 10) == list(range(10))
assert stran(x, 3, 10) == [20, 21, 22, 23, 24]
assert stran(x, 4, 10) == []
assert stevilo_strani(x, 10) == 3
assert opis_strani(x, 2, 10) == 'stran 2 od 3'
try:
    stran(x, 0, 10)
    raise SystemExit('ni ValueError')
except ValueError:
    pass''',
 [("    zacetek = st_strani * velikost\n", "    if st_strani < 1:\n        raise ValueError(\"stevilka strani mora biti vsaj 1\")\n    zacetek = (st_strani - 1) * velikost\n")]),

("konstanta", "orodje", "/projekt/ddv.py", DDV_MOD,
 "Preimenuj konstanto `DDV` v `STOPNJA_DDV` povsod v /projekt/ddv.py (`ZNIZAN_DDV` ostane, kot je).",
 r'''assert STOPNJA_DDV == 0.22 and ZNIZAN_DDV == 0.095
assert 'DDV' not in globals()
assert z_ddv(100) == 122.0 and z_ddv(100, True) == 109.5
assert brez_ddv(122) == 100.0
assert znesek_ddv(10) == 2.2
assert opis() == 'Splosna stopnja 22.0 %, znizana 9.5 %'
assert stopnje() == {'splosna': 0.22, 'znizana': 0.095}''',
 [("\nDDV = 0.22", "\nSTOPNJA_DDV = 0.22"), ("ZNIZAN_DDV if znizan else DDV\n    return round(neto", "ZNIZAN_DDV if znizan else STOPNJA_DDV\n    return round(neto"),
  ("ZNIZAN_DDV if znizan else DDV\n    return round(bruto", "ZNIZAN_DDV if znizan else STOPNJA_DDV\n    return round(bruto"),
  ("{DDV * 100", "{STOPNJA_DDV * 100"), ("\"splosna\": DDV,", "\"splosna\": STOPNJA_DDV,")]),

("komentarji", "orodje", "/project/ini.py", INI,
 "In /project/ini.py, parse() should also treat lines starting with ';' as comments, and a value may end with an inline comment that starts with ' #' or ' ;' which must be dropped (along with the spaces before it).",
 r'''t = "; top\n[a]\nx = 1 # one\ny=2\n# c\n[b]\nz = hello ; greeting\nw = a#b\n"
d = parse(t)
assert d == {'a': {'x': '1', 'y': '2'}, 'b': {'z': 'hello', 'w': 'a#b'}}, d
assert get(d, 'b', 'z') == 'hello' and get(d, 'b', 'q', 7) == 7
try:
    parse("[a]\nnoequals\n")
    raise SystemExit('no ValueError')
except ValueError:
    pass''',
 [("line.startswith(\"#\")", "line.startswith((\"#\", \";\"))"),
  ("        data[section][key.strip()] = value.strip()\n", "        for mark in (\" #\", \" ;\"):\n            if mark in value:\n                value = value[:value.index(mark)]\n        data[section][key.strip()] = value.strip()\n")]),

("sekunde", "orodje", "/projekt/cas.py", CAS,
 "Funkcija `preberi_cas` v /projekt/cas.py naj poleg oblike 'HH:MM' sprejme tudi 'HH:MM:SS' in vrne sekunde od polnoči. Sekunde morajo biti med 0 in 59, sicer ValueError.",
 r'''assert preberi_cas('01:02') == 3720
assert preberi_cas('01:02:03') == 3723
assert preberi_cas('23:59:59') == 86399
assert trajanje('10:00', '10:30:15') == 1815
assert zapisi_cas(3723) == '01:02:03'
for bad in ['24:00', '10:60', '10:00:60', '1:2:3:4']:
    try:
        preberi_cas(bad)
        raise SystemExit('ni ValueError za ' + bad)
    except ValueError:
        pass''',
 [("    ure, minute = niz.split(\":\")\n    ure, minute = int(ure), int(minute)\n    if not (0 <= ure < 24 and 0 <= minute < 60):\n        raise ValueError(f\"neveljaven cas: {niz}\")\n    return ure * 3600 + minute * 60\n",
   "    deli = niz.split(\":\")\n    if len(deli) not in (2, 3):\n        raise ValueError(f\"neveljaven cas: {niz}\")\n    ure, minute = int(deli[0]), int(deli[1])\n    sekunde = int(deli[2]) if len(deli) == 3 else 0\n    if not (0 <= ure < 24 and 0 <= minute < 60 and 0 <= sekunde < 60):\n        raise ValueError(f\"neveljaven cas: {niz}\")\n    return ure * 3600 + minute * 60 + sekunde\n")]),

("popust", "orodje", "/projekt/kosarica.py", KOSARICA,
 "Dodaj razredu `Kosarica` v /projekt/kosarica.py metodo `skupaj_z_popustom(odstotek)`, ki vrne skupno ceno, zmanjšano za podani odstotek, zaokroženo na 2 decimalki. Odstotek mora biti med 0 in 100 (oba vključno), sicer ValueError.",
 r'''k = Kosarica()
k.dodaj('kruh', 2.5, 2)
k.dodaj('mleko', 1.19)
assert k.skupaj() == 6.19
assert k.skupaj_z_popustom(10) == 5.57
assert k.skupaj_z_popustom(0) == 6.19
assert k.skupaj_z_popustom(100) == 0
assert k.stevilo_kosov() == 3
for bad in (-1, 100.5):
    try:
        k.skupaj_z_popustom(bad)
        raise SystemExit('ni ValueError')
    except ValueError:
        pass''',
 [("    def stevilo_kosov(self):\n", "    def skupaj_z_popustom(self, odstotek):\n        if not 0 <= odstotek <= 100:\n            raise ValueError(\"odstotek mora biti med 0 in 100\")\n        return round(self.skupaj() * (100 - odstotek) / 100, 2)\n\n    def stevilo_kosov(self):\n")]),

("imenik", "orodje", "/projekt/imenik.py", IMENIK,
 "Iskanje `Imenik.najdi` v /projekt/imenik.py naj ne loči velikih in malih črk ('ana' najde 'Ana'), `imena()` pa naj še vedno vrne imena, kot so bila vpisana.",
 r'''i = Imenik()
i.dodaj('Ana', '041 111 111')
i.dodaj('Bor', '031 222 222')
assert i.najdi('ana') == '041 111 111'
assert i.najdi('BOR') == '031 222 222'
assert i.najdi('Cene') is None
assert i.imena() == ['Ana', 'Bor']''',
 [("        return self._vnosi.get(ime)\n", "        for kljuc, stevilka in self._vnosi.items():\n            if kljuc.lower() == ime.lower():\n                return stevilka\n        return None\n")]),

("cela_sklad", "cela", "orodja.py", ORODJA,
 "Tu je datoteka orodja.py. V funkciji `simplify_path` preimenuj spremenljivko `stack` v `sklad`. Vrni celotno datoteko, sicer nespremenjeno, v enem bloku ```python.",
 _testi(*ORODJA_IMENA) + "\nimport inspect\n_s = inspect.getsource(simplify_path)\nassert 'sklad' in _s and 'stack' not in _s\nassert 'stack' in inspect.getsource(decode_string)",
 [("    stack = []\n    for part in path.split('/'):\n        if part in ('', '.'):\n            continue\n        if part == '..':\n            if stack:\n                stack.pop()\n        else:\n            stack.append(part)\n    return '/' + '/'.join(stack)",
   "    sklad = []\n    for part in path.split('/'):\n        if part in ('', '.'):\n            continue\n        if part == '..':\n            if sklad:\n                sklad.pop()\n        else:\n            sklad.append(part)\n    return '/' + '/'.join(sklad)")]),

("cela_sep", "cela", "orodja.py", ORODJA,
 "Here is orodja.py. Add an optional parameter `sep: str = '.'` to `compare_versions`, so that versions such as '1-2-10' can be compared with sep='-'. Return the whole file, otherwise unchanged, in one ```python block.",
 _testi(*ORODJA_IMENA) + "\nassert compare_versions('1-2', '1-10', sep='-') == -1\nassert compare_versions('2-0', '2', '-') == 0",
 [("def compare_versions(a, b):\n    x = [int(p) for p in a.split('.')]\n    y = [int(p) for p in b.split('.')]",
   "def compare_versions(a, b, sep='.'):\n    x = [int(p) for p in a.split(sep)]\n    y = [int(p) for p in b.split(sep)]")]),

("cela_razdeli", "cela", "racuni.py", RACUNI,
 "Tu je datoteka racuni.py. Funkcija `razdeli` pri enakih ostankih da cent zadnjemu namesto prvemu v seznamu. Popravi to in vrni celotno datoteko, sicer nespremenjeno, v enem bloku ```python.",
 _testi(*RACUNI_IMENA),
 [_RAZDELI_NAPAKA[::-1]]),

("cela_ime", "cela", "racuni.py", RACUNI.replace(*_RAZDELI_NAPAKA[::-1]),
 "Here is racuni.py. Rename the function `iban_valid` to `is_valid_iban` and return the whole file, otherwise unchanged, in one ```python block.",
 _testi(*RACUNI_IMENA).replace("iban_valid(", "is_valid_iban(") + "\nassert 'iban_valid' not in globals()",
 [("def iban_valid(s):", "def is_valid_iban(s):")]),
]


# English twins of the Slovenian tasks (2026-11-18; the user, for the fast profile: "slovenscina v tej verziji ni popolnoma nic
# pomembna vazno je pa da anglescina in koda dobro delata"): the same files, tests and references, the prompt in English. The
# verdict uses the English tasks (angleska()); the Slovenian ones stay for the record.
_F_EN = {
    "slug": "Write a Python function `slugify(besedilo: str) -> str` that makes a URL slug: all lowercase, č→c, š→s, ž→z, ć→c, đ→d (uppercase letters too), every run of characters that are not letters a-z or digits 0-9 becomes a single '-', and there is no '-' at the start or the end.",
    "davcna": "Write a Python function `davcna_veljavna(s: str) -> bool` for a Slovenian tax number: exactly 8 digits, the first one not 0. Multiply the first 7 digits by the weights 8, 7, 6, 5, 4, 3, 2 and add them up. The check digit is 11 - (sum mod 11); if that gives 10 the check digit is 0, if it gives 11 the number is invalid. The number is valid when its last digit equals the check digit. Spaces or any other characters make it invalid.",
    "razdeli": "Write a Python function `razdeli(znesek_centov: int, delezi: list[int]) -> list[int]` that splits an amount in cents in proportion to the shares (the shares add up to a positive number). Everyone first gets their share rounded down; the remaining cents go one each to those with the largest remainder of the division, and among equal remainders the one earlier in the list comes first. The result always adds up to the amount. Use integer arithmetic.",
    "delovni": "Write a Python function `delovni_dnevi(od: datetime.date, do: datetime.date, prazniki: set) -> int` that counts the working days (Monday to Friday, not in the set of holidays) from `od` to `do`, both inclusive. If `od` is after `do`, return 0. The dates may be several years apart.",
}
FUNKCIJE += [(fid + "_en", _F_EN[fid], tests, ref) for fid, _, tests, ref in list(FUNKCIJE) if fid in _F_EN]

_U_EN = {
    "zaloga": ("/project/zaloga.py", "In /project/zaloga.py, add a method `odvzemi(ime, kolicina)` to the Zaloga class that reduces the stock. If the quantity is not positive or there is not enough of the item, raise ValueError and leave the stock unchanged. An item that drops to 0 must disappear from `vsi()`."),
    "preimenuj": ("/project/besedilo.py", "Rename the function `stej_besede` to `prestej_besede` in /project/besedilo.py and fix all its calls."),
    "valuta": ("/project/cene.py", "In /project/cene.py, add an optional parameter `valuta` with the default '€' to `formatiraj_ceno` and use it instead of the fixed sign. `formatiraj_racun` must also accept an optional `valuta` (default '€') and pass it on."),
    "razvrsti": ("/project/studenti.py", "`razvrsti` in /project/studenti.py should sort the students by grade (`ocena`) descending, and by surname (`priimek`) ascending among equal grades (now it sorts by surname only)."),
    "strani": ("/project/strani.py", "Pages in /project/strani.py are numbered from 1, but `stran(x, 1, 10)` returns the second page. Fix that, and make a `st_strani` below 1 raise ValueError."),
    "konstanta": ("/project/ddv.py", "Rename the constant `DDV` to `STOPNJA_DDV` everywhere in /project/ddv.py (`ZNIZAN_DDV` stays as it is)."),
    "sekunde": ("/project/cas.py", "`preberi_cas` in /project/cas.py should accept 'HH:MM:SS' as well as 'HH:MM' and return the seconds since midnight. Seconds must be between 0 and 59, otherwise ValueError."),
    "popust": ("/project/kosarica.py", "Add a method `skupaj_z_popustom(odstotek)` to the `Kosarica` class in /project/kosarica.py that returns the total price reduced by the given percentage, rounded to 2 decimals. The percentage must be between 0 and 100 (both inclusive), otherwise ValueError."),
    "imenik": ("/project/imenik.py", "Make the lookup `Imenik.najdi` in /project/imenik.py case-insensitive ('ana' finds 'Ana'), while `imena()` still returns the names as they were entered."),
    "cela_sklad": ("orodja.py", "Here is orodja.py. In the function `simplify_path`, rename the variable `stack` to `sklad`. Return the whole file, otherwise unchanged, in one ```python block."),
    "cela_razdeli": ("racuni.py", "Here is racuni.py. When remainders are equal, the function `razdeli` gives the cent to the last item instead of the first one in the list. Fix it and return the whole file, otherwise unchanged, in one ```python block."),
}
UREJANJE += [(uid + "_en", mode, _U_EN[uid][0], original, _U_EN[uid][1], tests, pairs)
             for uid, mode, path, original, instr, tests, pairs in list(UREJANJE) if uid in _U_EN]
SLOVENSKE = set(_F_EN) | set(_U_EN)


def angleska(task_id):
    """True for a task in English (the fast profile's verdict counts these)."""
    return task_id not in SLOVENSKE


def numbered(text):
    """The read tool's output as Koda (OpenCode) shows it."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    body = "\n".join(f"{i:05d}| {line}" for i, line in enumerate(lines, 1))
    return f"<file>\n{body}\n\n(End of file - total {len(lines)} lines)\n</file>"


def apply_ref(original, pairs):
    text = original
    for old, new in pairs:
        assert text.count(old) == 1, (old[:60], text.count(old))
        text = text.replace(old, new)
    return text


def run_code(src, tests, timeout=20):
    """Runs src + tests in a fresh interpreter; (ok, error tail)."""
    import os, subprocess, sys, tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.py")
        open(p, "w", encoding="utf-8").write(src + "\n\n" + tests + "\nprint('OK')\n")
        try:
            r = subprocess.run([sys.executable, "-I", p], cwd=d, capture_output=True, text=True, timeout=timeout,
                               env={"PATH": "/usr/bin:/bin"})
            return r.returncode == 0 and r.stdout.strip().endswith("OK"), (r.stderr or r.stdout or "")[-400:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def code_of(answer):
    """The longest ```python block of an answer, or the whole answer when it has none."""
    import re
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", answer, re.S)
    return max(blocks, key=len) if blocks else answer
