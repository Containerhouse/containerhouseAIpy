# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015-2016, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

#We have adapted network_api_demo.py for our own research.
#We are only using the SensorRegion from the Network API. Encoded data is being fed to the latest
#spatial pooler, temporal memory, and SDR classifier algorithms. 
#The sample data is our own and is generated from our SketchUp ruby extension.

import copy
import csv
import json
import os

import numpy
from itertools import izip as zip, count 
from nupic.research.temporal_memory import TemporalMemory as TM

from pkg_resources import resource_filename
from nupic.engine import Network
from nupic.encoders import MultiEncoder, DeltaEncoder, ScalarEncoder, SDRCategoryEncoder
from nupic.data.file_record_stream import FileRecordStream

_VERBOSITY = 0  # how chatty the demo should be
_SEED = 1956  # the random seed used throughout
_INPUT_FILE_PATH = resource_filename(
  "nupic.datafiles", "extra/tm4.csv"
)
_OUTPUT_PATH = "point3d008-7out.csv"
_NUM_RECORDS = 7

tm = TM(columnDimensions = (3000,),
        cellsPerColumn=18,
        minThreshold=8,
        maxNewSynapseCount=20,
        permanenceIncrement=0.1,
        activationThreshold=8,
        seed=_SEED,
        )

def createEncoder():

 diagCoorA_encoder = ScalarEncoder(205, 0.0, 200.0, n=1000, name="diagCoorA")
 diagCoorB_encoder = ScalarEncoder(205, 0.0, 200.0, n=1000, name="diagCoorB")
 diagCoorC_encoder = ScalarEncoder(205, 0.0, 200.0, n=1000, name="diagCoorC") 

 global encoder 
 encoder = MultiEncoder()
 
 encoder.addEncoder("diagCoorA", diagCoorA_encoder)
 encoder.addEncoder("diagCoorB", diagCoorB_encoder)
 encoder.addEncoder("diagCoorC", diagCoorC_encoder)

 return encoder

def createNetwork(dataSource):

  network = Network()
  network.addRegion("sensor", "py.RecordSensor",
                    json.dumps({"verbosity": _VERBOSITY}))
  sensor = network.regions["sensor"].getSelf()
  sensor.encoder = createEncoder()
  sensor.dataSource = dataSource

  return network

def runNetwork(network, writer):
 
 sensorRegion = network.regions["sensor"]

 for h in xrange(_NUM_RECORDS):
 
    network.run(1)
   
    testGV = sensorRegion.getOutputData("dataOut").nonzero()[0] 
    src = sensorRegion.getOutputData("sourceOut")
    
    print src

    if h == 0:
     listDataOut = []
     listDataOut.insert(0,testGV)
    else: 
     listDataOut.append(testGV) 

 listCount = len(listDataOut)

 for j in xrange(listCount):

     num = int(j)
     mod = num % 2
     
     if mod == 0:

       if j < listCount-2:

        for k in xrange(100):

         tm.compute(listDataOut[j], learn = True) 
         tm.compute(listDataOut[j+1], learn = True) 

         tm.reset()

 tm.compute(listDataOut[listCount-1], learn=False)

 predictedColumnIndices = [tm.columnForCell(i) for i in tm.getPredictiveCells()]

 predArray = numpy.zeros(tm.numberOfColumns(), dtype="int64")
 predArray[list(predictedColumnIndices)] = 1

 testDecodeAlt = encoder.topDownCompute(predArray)

 for l in xrange(3):
  print "<>"
 print("predictions: " + str(testDecodeAlt[0][0]) + ", " + str(testDecodeAlt[1][0]) + ", " + str(testDecodeAlt[2][0]))
 for g in xrange(3):
  print "<>"

if __name__ == "__main__":
  
  dataSource = FileRecordStream(streamID=_INPUT_FILE_PATH)

  network = createNetwork(dataSource)
  network.initialize()
 
  outputPath = os.path.join(os.path.dirname(__file__), _OUTPUT_PATH)
  with open(outputPath, "w") as outputFile:
   writer = csv.writer(outputFile)
 
   runNetwork(network, writer)
