from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

from dds_data_structures import Point2D


@dataclass
@annotate.final
@annotate.autoid("sequential")
class Points2D(idl.IdlStruct, typename="Points2D"):
    points: types.sequence[Point2D]
