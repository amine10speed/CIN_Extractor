# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/region_similarity_calculator.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n)protos/region_similarity_calculator.proto\x12\x17object_detection.protos\"\xde\x02\n\x1aRegionSimilarityCalculator\x12N\n\x16neg_sq_dist_similarity\x18\x01 \x01(\x0b\x32,.object_detection.protos.NegSqDistSimilarityH\x00\x12@\n\x0eiou_similarity\x18\x02 \x01(\x0b\x32&.object_detection.protos.IouSimilarityH\x00\x12@\n\x0eioa_similarity\x18\x03 \x01(\x0b\x32&.object_detection.protos.IoaSimilarityH\x00\x12W\n\x1athresholded_iou_similarity\x18\x04 \x01(\x0b\x32\x31.object_detection.protos.ThresholdedIouSimilarityH\x00\x42\x13\n\x11region_similarity\"\x15\n\x13NegSqDistSimilarity\"\x0f\n\rIouSimilarity\"\x0f\n\rIoaSimilarity\"6\n\x18ThresholdedIouSimilarity\x12\x1a\n\riou_threshold\x18\x01 \x01(\x02:\x03\x30.5')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'protos.region_similarity_calculator_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _REGIONSIMILARITYCALCULATOR._serialized_start=71
  _REGIONSIMILARITYCALCULATOR._serialized_end=421
  _NEGSQDISTSIMILARITY._serialized_start=423
  _NEGSQDISTSIMILARITY._serialized_end=444
  _IOUSIMILARITY._serialized_start=446
  _IOUSIMILARITY._serialized_end=461
  _IOASIMILARITY._serialized_start=463
  _IOASIMILARITY._serialized_end=478
  _THRESHOLDEDIOUSIMILARITY._serialized_start=480
  _THRESHOLDEDIOUSIMILARITY._serialized_end=534
# @@protoc_insertion_point(module_scope)
