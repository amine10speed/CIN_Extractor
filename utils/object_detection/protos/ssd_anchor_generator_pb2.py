# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/ssd_anchor_generator.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n!protos/ssd_anchor_generator.proto\x12\x17object_detection.protos\"\xf2\x02\n\x12SsdAnchorGenerator\x12\x15\n\nnum_layers\x18\x01 \x01(\x05:\x01\x36\x12\x16\n\tmin_scale\x18\x02 \x01(\x02:\x03\x30.2\x12\x17\n\tmax_scale\x18\x03 \x01(\x02:\x04\x30.95\x12\x0e\n\x06scales\x18\x0c \x03(\x02\x12\x15\n\raspect_ratios\x18\x04 \x03(\x02\x12*\n\x1finterpolated_scale_aspect_ratio\x18\r \x01(\x02:\x01\x31\x12*\n\x1creduce_boxes_in_lowest_layer\x18\x05 \x01(\x08:\x04true\x12\x1d\n\x12\x62\x61se_anchor_height\x18\x06 \x01(\x02:\x01\x31\x12\x1c\n\x11\x62\x61se_anchor_width\x18\x07 \x01(\x02:\x01\x31\x12\x15\n\rheight_stride\x18\x08 \x03(\x05\x12\x14\n\x0cwidth_stride\x18\t \x03(\x05\x12\x15\n\rheight_offset\x18\n \x03(\x05\x12\x14\n\x0cwidth_offset\x18\x0b \x03(\x05')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'protos.ssd_anchor_generator_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _SSDANCHORGENERATOR._serialized_start=63
  _SSDANCHORGENERATOR._serialized_end=433
# @@protoc_insertion_point(module_scope)
