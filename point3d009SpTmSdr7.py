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

_VERBOSITY = 0  # how chatty the demo should be
_SEED = 1956  # the random seed used throughout
_INPUT_FILE_PATH = resource_filename(
  "nupic.datafiles", "extra/scanCsvGrid009SpTmSdr7long373.csv"
)
_OUTPUT_PATH = "point3d008-7out.csv"
_NUM_RECORDS = 67 

tm = TM(columnDimensions = (2048,),
        cellsPerColumn=18,
        initialPermanence=0.5,
        connectedPermanence=0.5,
        minThreshold=8,
        maxNewSynapseCount=20,
        permanenceIncrement=0.1,
        permanenceDecrement=0.0,
        activationThreshold=8,
        )

sp = SP(inputDimensions=(2325,),
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

sdr = SDR(steps=(3,),
          alpha=0.001,
          actValueAlpha=0.3,
          verbosity=0
          )

def createEncoder():
 volume_encoder = ScalarEncoder(21, 0.0, 20.0, n=200, name="volume", clipInput=False)
 floorheight_encoder = ScalarEncoder(21, 0.0, 24.0, n=125, name="floorheight", clipInput=False) 

 diagCoorA_encoder = ScalarEncoder(21, 0.0, 200.0, n=200, name="diagCoorA", clipInput=False)
 diagCoorB_encoder = ScalarEncoder(21, 0.0, 200.0, n=200, name="diagCoorB", clipInput=False)
 diagCoorC_encoder = ScalarEncoder(21, 0.0, 200.0, n=200, name="diagCoorC", clipInput=False)
 diagCoorD_encoder = ScalarEncoder(21, 0.0, 200.0, n=200, name="diagCoorD", clipInput=False)
 diagCoorE_encoder = ScalarEncoder(21, 0.0, 200.0, n=200, name="diagCoorE", clipInput=False)
 diagCoorF_encoder = ScalarEncoder(21, 0.0, 200.0, n=200, name="diagCoorF", clipInput=False)
 diagCoorG_encoder = ScalarEncoder(21, 0.0, 200.0, n=200, name="diagCoorG", clipInput=False)
 diagCoorH_encoder = ScalarEncoder(21, 0.0, 200.0, n=200, name="diagCoorH", clipInput=False)
 diagCoorI_encoder = ScalarEncoder(21, 0.0, 200.0, n=200, name="diagCoorI", clipInput=False)
 diagCoorJ_encoder = ScalarEncoder(21, 0.0, 200.0, n=200, name="diagCoorJ", clipInput=False) 

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

 for ii in xrange(_NUM_RECORDS):

    #print ii
 
    network.run(1)

    testGV = sensorRegion.getOutputData("dataOut")
    testGVsrc = sensorRegion.getOutputData("sourceOut")
 
    testArr = numpy.zeros(2048, dtype="int64")

    if ii < 66:

     sp.compute(testGV, 1, testArr)
 
     activeColumns = testArr.nonzero()[0]
     tm.compute(activeColumns, learn = True)
 
#    predCellsProto = tm.getPredictiveCells()
     actCellsProto = tm.getActiveCells()    
#    winnerCellsProto = tm.getWinnerCells() 

     for kk in xrange(12):

      classification={"bucketIdx": kk, "actValue": testGVsrc[kk]}
      #if actCellsProto: 
      sdr.compute(ii, actCellsProto, classification, 1, 0)  
      #if predCellsProto:
      # sdrLearn = sdr.compute(ii, predCellsProto, classification, 1, 0) 
      #sdrLearn = sdr.compute(ii, testArr, classification, 1, 0)
      #print sdrLearn


     num = int(ii)
     mod = num % 2
     if mod > 0:
      None #tm.reset()
     else:
      None



    elif ii == _NUM_RECORDS-1:

     print testGVsrc

     sp.compute(testGV, 1, testArr)
     
     activeColumns = testArr.nonzero()[0]
     tm.compute(activeColumns, learn = True)

     #predCells = tm.getPredictiveCells()
     actCells = tm.getActiveCells()

     #for p in xrange(12):
 
     classification = None 
   
     #predict = sdr.compute(ii+1, actCells, classification, 0, 1) 
     #predict = sdr.compute(ii, predCells, classification, 0, 1)
     predict = sdr.compute(ii, actCells, classification, 0, 1) 
     #predict = sdr.infer(predCells, classification) 
     #predict = sdr.infer(actCells, classification) 
     print predict


 #writer.writerow((ii,))


if __name__ == "__main__":
  
  dataSource = FileRecordStream(streamID=_INPUT_FILE_PATH)

  network = createNetwork(dataSource)
  network.initialize()
 
  outputPath = os.path.join(os.path.dirname(__file__), _OUTPUT_PATH)
  with open(outputPath, "w") as outputFile:
   writer = csv.writer(outputFile)
 
   runNetwork(network, writer)
