# SPDX-FileCopyrightText: 2023 Philippe Proulx <eeppeliteloop@gmail.com>
# SPDX-License-Identifier: MIT

import normand


def test_init_labels():
    labels = {"yo": 0x88, "meow": 123}  # type: normand.LabelsT
    res = normand.parse("11 22 [yo:8] 33", init_labels=labels.copy())
    assert res.data == bytearray([0x11, 0x22, 0x88, 0x33])
    assert res.labels == labels.copy()


def test_init_vars():
    variables = {"zoom": 0x88, "bateau": -123.45}  # type: normand.VariablesT
    res = normand.parse("11 22 [zoom:8] 33", init_variables=variables.copy())
    assert res.data == bytearray([0x11, 0x22, 0x88, 0x33])
    assert res.variables == variables.copy()


def test_init_offset():
    res = normand.parse("11 22 [ICITTE:8] 33", init_offset=0x23)
    assert res.data == bytearray([0x11, 0x22, 0x25, 0x33])
    assert res.offset == 0x27


def _test_init_bo(bo: normand.ByteOrder):
    h_byte = 0xF8
    l_byte = 0x37

    if bo == normand.ByteOrder.LE:
        h_byte, l_byte = l_byte, h_byte

    res = normand.parse("11 22 [-1993:16] 33", init_byte_order=bo)
    assert res.data == bytearray([0x11, 0x22, h_byte, l_byte, 0x33])
    assert res.byte_order == bo


def test_init_bo_be():
    _test_init_bo(normand.ByteOrder.BE)


def test_init_bo_le():
    _test_init_bo(normand.ByteOrder.LE)


def test_final_labels():
    labels = {"yo": 0x88, "meow": 123}  # type: normand.LabelsT
    res = normand.parse(
        "11 <june> 22 (77 <aug> 88) * 2 <kilo> 33", init_labels=labels.copy()
    )
    labels["june"] = 1
    labels["kilo"] = 6
    assert res.labels == labels.copy()


def test_final_vars():
    variables = {"yo": 0x88, "meow": -123.45}
    res = normand.parse(
        "11 {yo = 18.2} 22 (77 {zoom = ICITTE} 88) * 2 33",
        init_variables=variables.copy(),
    )
    variables["yo"] = 18.2
    variables["zoom"] = 5
    assert res.variables == variables.copy()


def test_final_offset():
    res = normand.parse("11 22 33 <32> 44 55")
    assert res.offset == 34


def test_final_byte_order_none():
    res = normand.parse("11 22 33 [-19:8] 44 55")
    assert res.byte_order is None


def test_final_byte_order_be():
    res = normand.parse("11 22 !le 33 [-19:8] 44 ( !be 88 ) * 3 55")
    assert res.byte_order == normand.ByteOrder.BE


def test_final_byte_order_le():
    res = normand.parse("11 22 !be 33 [-19:8] 44 ( !le 88 ) * 3 55")
    assert res.byte_order == normand.ByteOrder.LE


def test_no_data():
    res = normand.parse("# nothing to see here!\n")
    assert len(res.data) == 0
    assert len(res.labels) == 0
    assert len(res.variables) == 0
    assert res.offset == 0
