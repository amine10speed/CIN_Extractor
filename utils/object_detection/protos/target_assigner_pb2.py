# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/target_assigner.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from protos import box_coder_pb2 as protos_dot_box__coder__pb2
from protos import matcher_pb2 as protos_dot_matcher__pb2
from protos import region_similarity_calculator_pb2 as protos_dot_region__similarity__calculator__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1cprotos/target_assigner.proto\x12\x17object_detection.protos\x1a\x16protos/box_coder.proto\x1a\x14protos/matcher.proto\x1a)protos/region_similarity_calculator.proto\"\xcd\x01\n\x0eTargetAssigner\x12\x31\n\x07matcher\x18\x01 \x01(\x0b\x32 .object_detection.protos.Matcher\x12R\n\x15similarity_calculator\x18\x02 \x01(\x0b\x32\x33.object_detection.protos.RegionSimilarityCalculator\x12\x34\n\tbox_coder\x18\x03 \x01(\x0b\x32!.object_detection.protos.BoxCoder')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'protos.target_assigner_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _TARGETASSIGNER._serialized_start=147
  _TARGETASSIGNER._serialized_end=352
# @@protoc_insertion_point(module_scope)
