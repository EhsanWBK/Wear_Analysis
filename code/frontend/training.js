"use strict";

// =============================================
// 	  Model Training - Widget Initialization			
// =============================================

const buttonTrainModel = document.getElementById('trainModelButton');
const buttonSaveModel = document.getElementById('saveModel');

const tableAugmentation = document.getElementById('augmentTable');

// Needed when more model architectures:
const checkModelVersion1 = document.getElementById('modelVersion1');
const checkModelVersion2 = document.getElementById('modelVersion2');

const checkRandomSelection = document.getElementById('randomSelection');
const checkSelectAugmentation = document.getElementById('selectAug');
const checkHorizontalFlip = document.getElementById('horizontalFlip');
const checkVerticalFlip = document.getElementById('verticalFlip');
const checkTransferLearn = document.getElementById('transferLearn');
const checkShuffleTrain = document.getElementById('shuffleTrain');

const inputModelName = document.getElementById('modelName');
const inputModelSaveDir = document.getElementById('modelSavingDir');
const inputProjectDir = document.getElementById('trainingImgDir');
const inputImageHeigth = document.getElementById('imageHeight');
const inputImageWidth = document.getElementById('imageWidth');
const inputNrChannels = document.getElementById('nrChannels');
const inputValSize = document.getElementById('validationSize');
const inputRandomState = document.getElementById('randomState');
const inputRotationRange = document.getElementById('rotationRange');
const inputWidthShiftRange = document.getElementById('widthShiftRange');
const inputHeightShiftRange = document.getElementById('heigthShiftRange');
const inputZoomRange = document.getElementById('zoomRange');
const inputBatchSize = document.getElementById('batchSize');
const inputNrEpochs = document.getElementById('nrEpochs');
const inputEarlyStopping = document.getElementById('earlyStopping');

// =========================================
// 	  Model Training - HTML Functions			
// =========================================

buttonSaveModel.disabled = true;

function augmentationSelected() {
	if (checkSelectAugmentation.checked){ tableAugmentation.style.display = 'block'; }
	if (checkSelectAugmentation.checked==false){ tableAugmentation.style.display = 'none'; }
}

function selectModelVersion() {
	if (checkModelVersion1.checked){
		checkModelVersion2.cheked = false;
	}
	if (checkModelVersion2.checked){
		checkModelVersion1.cheked = false;
	}
}

// =========================================
// 	  Model Training - Python Functions			
// =========================================

function py_train() {

	var parDict = {};

	parDict['modelName'] = inputModelName.value;
	parDict['modelSavingDir'] = inputModelSaveDir.value;
	parDict['trainingImgDir'] = inputProjectDir.value;
	parDict['imageHeight'] = inputImageHeigth.value;
	parDict['imageWidth'] = inputImageWidth.value;
	parDict['nrChannels'] = inputNrChannels.value;
	parDict['validationSize'] = inputValSize.value;
	parDict['randomState'] = inputRandomState.value;
	parDict['randomSelection'] = checkRandomSelection.checked;
	parDict['batchSize'] = inputBatchSize.value;
	parDict['nrEpochs'] = inputNrEpochs.value;
	parDict['earlyStopping'] = inputEarlyStopping.value;
	parDict['transferLearn'] = checkTransferLearn.checked;
	parDict['shuffleTrain'] = checkShuffleTrain.checked;
	parDict['selectAug'] = checkSelectAugmentation.checked;
	parDict['rotationRange'] = inputRotationRange.value;
	parDict['widthShiftRange'] = inputWidthShiftRange.value;
	parDict['heigthShiftRange'] = inputHeightShiftRange.value;
	parDict['zoomRange'] = inputZoomRange.value;
	parDict['horizontalFlip'] = checkHorizontalFlip.checked;
	parDict['verticalFlip'] = checkVerticalFlip.checked;;

	console.table(parDict);
	eel.trainModel(parDict)();
}

function py_saveModel(){ eel.saveModel(document.getElementById('modelSavingDir')); }
	
eel.expose(displayTensorboard);
function displayTensorboard(url) {
	window.open(url,"_blank").focus();
}

eel.expose(modelTrained);
function modelTrained() {
	if (checkToTensorBoard.checked){ displayTensorboard(); }
	buttonSaveModel.disabled = false;
}