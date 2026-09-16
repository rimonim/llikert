import json
import struct

import pytest

from llikert.server.canonical import _number, canonicalize, identity_hash

# RFC 8785 appendix B: IEEE-754 bit patterns and their canonical text
RFC8785_NUMBERS = [
    ("0000000000000000", "0"),
    ("8000000000000000", "0"),
    ("0000000000000001", "5e-324"),
    ("8000000000000001", "-5e-324"),
    ("7fefffffffffffff", "1.7976931348623157e+308"),
    ("ffefffffffffffff", "-1.7976931348623157e+308"),
    ("4340000000000000", "9007199254740992"),
    ("c340000000000000", "-9007199254740992"),
    ("4430000000000000", "295147905179352830000"),
    ("44b52d02c7e14af5", "9.999999999999997e+22"),
    ("44b52d02c7e14af6", "1e+23"),
    ("44b52d02c7e14af7", "1.0000000000000001e+23"),
    ("444b1ae4d6e2ef4e", "999999999999999700000"),
    ("444b1ae4d6e2ef4f", "999999999999999900000"),
    ("444b1ae4d6e2ef50", "1e+21"),
    ("3eb0c6f7a0b5ed8c", "9.999999999999997e-7"),
    ("3eb0c6f7a0b5ed8d", "0.000001"),
    ("3eb0c6f7a0b5ed8e", "0.0000010000000000000002"),
    ("41b3de4355555553", "333333333.3333332"),
    ("41b3de4355555554", "333333333.33333325"),
    ("41b3de4355555555", "333333333.3333333"),
    ("41b3de4355555556", "333333333.3333334"),
    ("41b3de4355555557", "333333333.33333343"),
    ("becbf647612f3696", "-0.0000033333333333333333"),
    ("43143ff3c1cb0959", "1424953923781206.2"),
]


@pytest.mark.parametrize("bits,expected", RFC8785_NUMBERS)
def test_rfc8785_numbers(bits, expected):
    (x,) = struct.unpack(">d", bytes.fromhex(bits))
    assert _number(x) == expected


def test_small_values():
    assert _number(1) == "1"
    assert _number(1.0) == "1"
    assert _number(-2.5) == "-2.5"
    assert _number(0.1) == "0.1"
    assert _number(1e-7) == "1e-7"
    assert _number(1.5e-7) == "1.5e-7"
    assert _number(123e18) == "123000000000000000000"


def test_nonfinite_and_bool_rejected():
    with pytest.raises(ValueError):
        _number(float("nan"))
    with pytest.raises(ValueError):
        _number(float("inf"))
    with pytest.raises(TypeError):
        canonicalize({"x": object()})


BS = chr(92)  # backslash
DQ = chr(34)  # double quote


def test_rfc8785_structure_example():
    string_value = chr(0x20AC) + "$" + chr(0x0F) + chr(0x0A) + "A'" + chr(0x42) + DQ + BS + BS + DQ + "/"
    value = {
        "numbers": [333333333.33333329, 1e30, 4.50, 2e-3, 0.000000000000000000000000001],
        "string": string_value,
        "literals": [None, True, False],
    }
    expected_string = (
        DQ + chr(0x20AC) + "$" + BS + "u000f" + BS + "n" + "A'B" + BS + DQ + BS + BS + BS + BS + BS + DQ + "/" + DQ
    )
    expected = (
        '{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],'
        '"string":' + expected_string + "}"
    )
    assert canonicalize(value) == expected


def test_rfc8785_key_order_utf16():
    keys = [chr(0x20AC), chr(0x0D), chr(0xFB33), "1", chr(0x1F600), chr(0x80), chr(0xF6)]
    value = {k: i for i, k in enumerate(keys)}
    keys_in_order = [chr(0x0D), "1", chr(0x80), chr(0xF6), chr(0x20AC), chr(0x1F600), chr(0xFB33)]
    out = canonicalize(value)
    positions = [out.index(canonicalize(k) + ":") for k in keys_in_order]
    assert positions == sorted(positions)


def test_identity_hash_ignores_input_key_order():
    assert identity_hash({"a": 1, "b": [1, 2]}) == identity_hash({"b": [1, 2], "a": 1})
    assert identity_hash({"a": 1}).startswith("sha256:")


def test_numbers_match_ecmascript_fixture(fixtures):
    data = json.loads((fixtures / "canonical" / "numbers-es.json").read_text())
    mismatches = []
    for bits, expected in data["cases"]:
        (x,) = struct.unpack(">d", bytes.fromhex(bits))
        if _number(x) != expected:
            mismatches.append((bits, _number(x), expected))
    assert not mismatches[:5]
    assert len(data["cases"]) > 2500
