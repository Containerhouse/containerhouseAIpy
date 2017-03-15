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
from nupic.research.spatial_pooler import SpatialPooler as SP
from nupic.algorithms.sdr_classifier import SDRClassifier as SDR

from pkg_resources import resource_filename
from nupic.engine import Network
from nupic.encoders import MultiEncoder, ScalarEncoder
from nupic.data.file_record_stream import FileRecordStream

_VERBOSITY = 0
_SEED = 1956
_INPUT_FILE_PATH = resource_filename(
  "nupic.datafiles", "extra/tm5.csv"
)
_OUTPUT_PATH = "point3d008-7out.csv"
_NUM_RECORDS = 9

tm = TM(columnDimensions = (2048,),
        cellsPerColumn=18,
        minThreshold=8,
        maxNewSynapseCount=20,
        permanenceIncrement=0.1,
        activationThreshold=8,
        seed=_SEED,
        )

sp = SP(inputDimensions=(165,),
        columnDimensions=(2048,),
        potentialRadius=16,
        potentialPct=0.5,
        globalInhibition=True,
        localAreaDensity=-1.0,
        numActiveColumnsPerInhArea=10.0,
        stimulusThreshold=0,
        synPermInactiveDec=0.008,
        synPermActiveInc=0.05,
        synPermConnected=0.10,
        minPctOverlapDutyCycle=0.001,
        dutyCyclePeriod=1000,
        maxBoost=20.0,
        seed=-1,
        spVerbosity=0,
        wrapAround=True
        )

sdr = SDR(steps=(1,),
          alpha=0.001,
          actValueAlpha=0.3,
          verbosity=0
          )

def createEncoder():

 diagCoorA_encoder = ScalarEncoder(55, 0.0, 200.0, n=200, name="diagCoorA")
 diagCoorB_encoder = ScalarEncoder(55, 0.0, 200.0, n=200, name="diagCoorB")
 diagCoorC_encoder = ScalarEncoder(55, 0.0, 200.0, n=200, name="diagCoorC") 

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
  
 listDataOut = []
 srcDataOut = []

 for h in xrange(_NUM_RECORDS):
 
    network.run(1)
   
    testGV = sensorRegion.getOutputData("dataOut").nonzero()[0] 
    listDataOut.append(testGV) 

    src = sensorRegion.getOutputData("sourceOut")
    print src
    tri = {} 
    tri[0] = src[0]
    tri[1] = src[1]
    tri[2] = src[2] 
    srcDataOut.append(tri)

 listCount = len(listDataOut)

 for j in xrange(listCount):

     num = int(j)
     mod = num % 2
     
     if mod == 0:

       if j < listCount-2:

        for k in xrange(200):

         testArr = numpy.zeros(2048, dtype="int64")	

         sp.compute(listDataOut[j], 1, testArr)	
         activeColumns = testArr.nonzero()[0]
         tm.compute(activeColumns, learn = True)

         actCellsProto = tm.getActiveCells() 

         for kk in xrange(3):

	  classification={"bucketIdx": kk+1, "actValue": srcDataOut[j][kk]}
          sdr.compute(j+1, actCellsProto, classification, 1, 0) 


         testArr2 = numpy.zeros(2048, dtype="int64")	

         sp.compute(listDataOut[j+1], 1, testArr2)	
         activeColumns2 = testArr2.nonzero()[0]
         tm.compute(activeColumns2, learn = True)

         actCellsProto2 = tm.getActiveCells() 

         for mm in xrange(3):

	  classification2={"bucketIdx": mm+1, "actValue": srcDataOut[j+1][mm]}
          sdr.compute(j+2, actCellsProto2, classification2, 1, 0) 


         tm.reset()

 testArr3 = numpy.zeros(2048, dtype="int64")	
 
 sp.compute(listDataOut[listCount-1], 0, testArr3)	
 activeColumns3 = testArr3.nonzero()[0]

 tm.compute(activeColumns3, learn = False)

 actCellsProto3 = tm.getActiveCells() 

# classification3 = None 
 for oo in xrange(3):
  classification3={"bucketIdx": oo+1, "actValue": srcDataOut[listCount-1][oo]}
 #classification3 = {"bucketIdx": 3, "actValue": srcDataOut[8][2]} 
  predict = sdr.compute(listCount, actCellsProto3, classification3, 0, 1) 
  print predict


if __name__ == "__main__":
  
  dataSource = FileRecordStream(streamID=_INPUT_FILE_PATH)

  network = createNetwork(dataSource)
  network.initialize()
 
  outputPath = os.path.join(os.path.dirname(__file__), _OUTPUT_PATH)
  with open(outputPath, "w") as outputFile:
   writer = csv.writer(outputFile)
 
   runNetwork(network, writer)
