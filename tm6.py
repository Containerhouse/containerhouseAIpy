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
from nupic.encoders import MultiEncoder, ScalarEncoder
from nupic.data.file_record_stream import FileRecordStream

_VERBOSITY = 0  # how chatty the demo should be
_SEED = 1956  # the random seed used throughout
_INPUT_FILE_PATH = resource_filename(
  "nupic.datafiles", "extra/tm6.csv"
)
_OUTPUT_PATH = "tm6out.csv"
_NUM_RECORDS = 9

tm = TM(columnDimensions = (10225,),
        cellsPerColumn=18,
        minThreshold=8,
        maxNewSynapseCount=20,
        permanenceIncrement=0.1,
        activationThreshold=8,
        seed=_SEED,
        )

def createEncoder():

 volume_encoder = ScalarEncoder(7, 0.0, 70.0, n=200, name="volume", forced=True)  
 floorheight_encoder = ScalarEncoder(1, 0.0, 70.0, n=25, name="floorheight", forced=True) 
 diagCoorA_encoder = ScalarEncoder(105, 0.0, 200.0, n=1000, name="diagCoorA")
 diagCoorB_encoder = ScalarEncoder(105, 0.0, 200.0, n=1000, name="diagCoorB")
 diagCoorC_encoder = ScalarEncoder(105, 0.0, 200.0, n=1000, name="diagCoorC") 
 diagCoorD_encoder = ScalarEncoder(105, 0.0, 200.0, n=1000, name="diagCoorD")
 diagCoorE_encoder = ScalarEncoder(105, 0.0, 200.0, n=1000, name="diagCoorE")
 diagCoorF_encoder = ScalarEncoder(105, 0.0, 200.0, n=1000, name="diagCoorF")
 diagCoorG_encoder = ScalarEncoder(105, 0.0, 200.0, n=1000, name="diagCoorG")
 diagCoorH_encoder = ScalarEncoder(105, 0.0, 200.0, n=1000, name="diagCoorH")
 diagCoorI_encoder = ScalarEncoder(105, 0.0, 200.0, n=1000, name="diagCoorI")
 diagCoorJ_encoder = ScalarEncoder(105, 0.0, 200.0, n=1000, name="diagCoorJ")

 global encoder 
 encoder = MultiEncoder()
 
 encoder.addEncoder("volume", volume_encoder) 
 encoder.addEncoder("floorheight", floorheight_encoder)
 encoder.addEncoder("diagCoorA", diagCoorA_encoder)
 encoder.addEncoder("diagCoorB", diagCoorB_encoder)
 encoder.addEncoder("diagCoorC", diagCoorC_encoder)
 encoder.addEncoder("diagCoorD", diagCoorD_encoder)
 encoder.addEncoder("diagCoorE", diagCoorE_encoder)
 encoder.addEncoder("diagCoorF", diagCoorF_encoder)
 encoder.addEncoder("diagCoorG", diagCoorG_encoder)
 encoder.addEncoder("diagCoorH", diagCoorH_encoder)
 encoder.addEncoder("diagCoorI", diagCoorI_encoder)
 encoder.addEncoder("diagCoorJ", diagCoorJ_encoder)

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

 listDataOut = []

 for h in xrange(_NUM_RECORDS):

    network.run(1)

    testGV = sensorRegion.getOutputData("dataOut").nonzero()[0] 
    listDataOut.append(testGV) 

    src = sensorRegion.getOutputData("sourceOut")
    if h == 3:
     print "<<<>>>"
     print "should predict:"
     for hSub in xrange(len(src)):
      if src[hSub] != 0.0:
       print src[hSub]

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

 print "predictions:"
 for l in xrange(len(testDecodeAlt)):
  if testDecodeAlt[l][0] != 0.0:
   print testDecodeAlt[l][0] 

if __name__ == "__main__":
  
  dataSource = FileRecordStream(streamID=_INPUT_FILE_PATH)

  network = createNetwork(dataSource)
  network.initialize()
 
  outputPath = os.path.join(os.path.dirname(__file__), _OUTPUT_PATH)
  with open(outputPath, "w") as outputFile:
   writer = csv.writer(outputFile)
 
   runNetwork(network, writer)
