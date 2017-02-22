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
  "nupic.datafiles", "extra/tm3-1.csv"
)
_OUTPUT_PATH = "point3d008-7out.csv"
_NUM_RECORDS = 5

tm = TM(columnDimensions = (2048,),
        #cellsPerColumn=32, #18 
        #initialPermanence=0.5,
        #connectedPermanence=0.5,
        minThreshold=2, #5, #8,
        maxNewSynapseCount=20,
        permanenceIncrement=0.1,
        #permanenceDecrement=0.0,
        activationThreshold=2, #5, #8,
        seed=_SEED,
        )

def createEncoder():
 #volume_encoder = ScalarEncoder(7, 0.0, 70.0, n=200, name="volume", clipInput=False, forced=True)
 #floorheight_encoder = ScalarEncoder(1, 0.0, 70.0, n=25, name="floorheight", clipInput=False, forced=True) 

 diagCoorA_encoder = ScalarEncoder(257, 0.0, 200.0, n=2048, name="diagCoorA", clipInput=False, forced=True)
 #diagCoorB_encoder = ScalarEncoder(157, 0.0, 200.0, n=2000, name="diagCoorB", clipInput=False, forced=True)
 #diagCoorC_encoder = ScalarEncoder(157, 0.0, 200.0, n=2000, name="diagCoorC", clipInput=False, forced=True)
 #diagCoorD_encoder = ScalarEncoder(157, 0.0, 200.0, n=2000, name="diagCoorD", clipInput=False, forced=True)
 #diagCoorE_encoder = ScalarEncoder(157, 0.0, 200.0, n=2000, name="diagCoorE", clipInput=False, forced=True)
 #diagCoorF_encoder = ScalarEncoder(157, 0.0, 200.0, n=2000, name="diagCoorF", clipInput=False, forced=True)
 #diagCoorG_encoder = ScalarEncoder(157, 0.0, 200.0, n=2000, name="diagCoorG", clipInput=False, forced=True)
 #diagCoorH_encoder = ScalarEncoder(157, 0.0, 200.0, n=2000, name="diagCoorH", clipInput=False, forced=True)
 #diagCoorI_encoder = ScalarEncoder(157, 0.0, 200.0, n=2000, name="diagCoorI", clipInput=False, forced=True)
 #diagCoorJ_encoder = ScalarEncoder(157, 0.0, 200.0, n=2000, name="diagCoorJ", clipInput=False, forced=True) 

 global encoder 
 encoder = MultiEncoder()
 
 #encoder.addEncoder("volume", volume_encoder)
 #encoder.addEncoder("floorheight", floorheight_encoder)
 encoder.addEncoder("diagCoorA", diagCoorA_encoder)
 #encoder.addEncoder("diagCoorB", diagCoorB_encoder)
 #encoder.addEncoder("diagCoorC", diagCoorC_encoder)
 #encoder.addEncoder("diagCoorD", diagCoorD_encoder)
 #encoder.addEncoder("diagCoorE", diagCoorE_encoder)
 #encoder.addEncoder("diagCoorF", diagCoorF_encoder)
 #encoder.addEncoder("diagCoorG", diagCoorG_encoder)
 #encoder.addEncoder("diagCoorH", diagCoorH_encoder)
 #encoder.addEncoder("diagCoorI", diagCoorI_encoder)
 #encoder.addEncoder("diagCoorJ", diagCoorJ_encoder)

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

 for ii in xrange(_NUM_RECORDS):
 
    network.run(1)
   
    testGV = sensorRegion.getOutputData("dataOut").nonzero()[0] 
    #print testGV
    src = sensorRegion.getOutputData("sourceOut")
    print src
    
    if ii == 0:
        zero = testGV
    elif ii == 1:
        one = testGV
    elif ii == 2:
        two = testGV
    elif ii == 3:
        three = testGV
    elif ii == 4:
        four = testGV
 
 for jj in xrange(100):

    #print jj

    tm.compute(zero, learn = True) 
    #tm.activateCells(zero, learn=True)
    #tm.activateDendrites(learn=True)

    tm.compute(one, learn = True) 
    #tm.activateCells(one, learn=True)
    #tm.activateDendrites(learn=True)
    tm.reset()

 #tm.reset()

 for jj in xrange(100):

    #print jj
    
    tm.compute(two, learn = True)
    #tm.activateCells(two, learn=True)
    #tm.activateDendrites(learn=True)
    
    tm.compute(three, learn = True)
    #tm.activateCells(three, learn=True)
    #tm.activateDendrites(learn=True)
    tm.reset()
 #tm.reset()
 #for mm in xrange(20): 
 # print mm
 tm.compute(four, learn=False)
 #tm.activateCells(four, learn=False)
 #tm.activateDendrites(learn=False)

 predictedColumnIndices = [tm.columnForCell(i) for i in tm.getPredictiveCells()]

 predArray = numpy.zeros(tm.numberOfColumns(), dtype="int64")
 predArray[list(predictedColumnIndices)] = 1

 testDecodeAlt = encoder.topDownCompute(predArray)

 for f in xrange(3):
  print "<>"
 print "prediction:" 
 print testDecodeAlt[0][0] #,testDecodeAlt[1][0],testDecodeAlt[2][0],testDecodeAlt[3][0],testDecodeAlt[4][0],testDecodeAlt[5][0],testDecodeAlt[6][0],testDecodeAlt[7][0],testDecodeAlt[8][0],testDecodeAlt[9][0],testDecodeAlt[10][0], testDecodeAlt[11][0] #testDecodeAlt
 for g in xrange(3):
  print "<>"
 #writer.writerow((ii,testDecodeAlt[0][0],testDecodeAlt[1][0],testDecodeAlt[2][0],testDecodeAlt[3][0],testDecodeAlt[4][0],testDecodeAlt[5][0],testDecodeAlt[6][0],testDecodeAlt[7][0],testDecodeAlt[8][0],testDecodeAlt[9][0],testDecodeAlt[10][0], testDecodeAlt[11][0]))

if __name__ == "__main__":
  
  dataSource = FileRecordStream(streamID=_INPUT_FILE_PATH)

  network = createNetwork(dataSource)
  network.initialize()
 
  outputPath = os.path.join(os.path.dirname(__file__), _OUTPUT_PATH)
  with open(outputPath, "w") as outputFile:
   writer = csv.writer(outputFile)
 
   runNetwork(network, writer)
