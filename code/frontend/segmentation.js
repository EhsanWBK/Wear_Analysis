"use strict";

// Canvas Display
const canvas1 = document.getElementById('screen1')
const canvas2 = document.getElementById('screen2')

// Segmentation
const buttonEnableCam = document.getElementById('enableCam');
const imageFileInput = document.getElementById('imageInput');

// Hidden Elements
document.getElementById('segmentImage').disabled = true;

// ===========================================
// 	  Image Segmentation - HTML Functions			
// ===========================================

function setSingleImage(){
	document.getElementById('stackData').checked = false;
	if (document.getElementById('enableCam').checked){
		document.getElementById('runRoutine').style.display = 'none';
		document.getElementById('takePicture').style.display = 'block';
	}
	else{document.getElementById('nrEdgesTable').style.display = 'none';}
}

function setDataStack(){
	document.getElementById('singleImage').checked = false;
	if (document.getElementById('enableCam').checked){
		document.getElementById('takePicture').style.display = 'none';
		document.getElementById('runRoutine').style.display = 'block';
		
	}
	else{document.getElementById('nrEdgesTable').style.display = 'block';}
}

function getData(elementId){
	if (document.getElementById('stackData').checked){ py_directory(elementId);}
	else{ py_file(elementId);}
	let model = document.getElementById('modelDirPath').value;
	// if (!(model == null) && !(model =="")){ document.getElementById('segmentImage').disabled = true;}
}

// ============================================
// 	  Image Segmentation - Python Functions			
// ============================================

eel.expose(updateCanvas1);
function updateCanvas1(val) { screen1.src = "data:image/jpeg;base64," + val; }

eel.expose(updateCanvas2);
function updateCanvas2(val) { screen2.src = "data:image/jpeg;base64," + val }

function py_video() {
	// Hide Buttons
	if (document.getElementById('enableCam').checked){
		// Starting Video Stream and displaying on Canvas 1 (done in python backend and updateCanvas1())
		eel.videoFeed()()
		// Set and disable Buttons
		document.getElementById('dataSelection').disabled = true;
		document.getElementById('dataSelection').style.display = 'none';
		document.getElementById('imageInput').type = "hidden";
		document.getElementById('nrEdgesTable').style.display = 'none';
		if (document.getElementById('singleImage').checked){ document.getElementById('takePicture').style.display = 'block'; } 
		else if (document.getElementById('singleImage').checked==false){ document.getElementById('runRoutine').style.display = 'block'; }
	} 
	if (document.getElementById('enableCam').checked==false){
		// Stopping Video Stream
		eel.stopVideo()()
		document.getElementById('dataSelection').disabled = false;
		document.getElementById('dataSelection').style.display = 'block';
		document.getElementById('imageInput').type = "text";
		// Set and disable Buttons
		document.getElementById('takePicture').style.display = 'none';
		document.getElementById('runRoutine').style.display = 'none';
		document.getElementById('nrEdgesTable').style.display = 'none';
	}	
}

function py_foto() { 
	// Take picture from Video Stream and displaying on Canvas 1 (done in python backend and updateCanvas1())
	eel.takePicture()();
	// Stopping and resetting Video Stream 
	if (document.getElementById('enableCam').checked){
		document.getElementById('enableCam').checked = false;
		py_video();
	}
 }

function py_videoSegment() {
	// Segment Video and Display Live Segmentation on Canvas 2 (done in python backend and updateCanvas2())
	eel.segmentVideo()();
}

function py_segment() { 
	// Segment Single Image or Image Stack and displaying results on Canvas 2 (done in python backend and updateCanvas2())
	if (document.getElementById('singleImage').checked){eel.segmentImage()(); };
	if (document.getElementById('stackData').checked){
		eel.segmentStack(document.getElementById('imageInput').value, document.getElementById('nrEdgesInput').value)(); };
}

function py_loadModel(elementId) {
	// Load Model from Directory
	py_file(elementId);
	document.getElementById('segmentImage').disabled = false;
}

function py_saveResults() {
	// Save Segmentation Results to directory
	eel.saveSegResult(document.getElementById('saveResultDir').value)();
}
