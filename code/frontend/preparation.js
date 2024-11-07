"use strict";

// =========================================
// 	  Data Preparation - HTML Functions			
// =========================================

function resizeSelect() {
	if (document.getElementById('resizeEnableBut').checked==false){
		document.getElementById('heigthLabel').style.display = 'none';
		document.getElementById('widthLabel').style.display = 'none';
		document.getElementById('aspectRatioLabel').style.display = 'none';
		document.getElementById('aspectRatioHeight').type = "hidden";
		document.getElementById('aspectRatioWidth').type = "hidden";
		document.getElementById('resizeButton').style.display = 'none';
		document.getElementById('explainResize').style.display = 'none';
	}
	if (document.getElementById('resizeEnableBut').checked){
		document.getElementById('heigthLabel').style.display = 'block';
		document.getElementById('widthLabel').style.display = 'block';
		document.getElementById('aspectRatioLabel').style.display = 'block';
		document.getElementById('aspectRatioHeight').type = "text";
		document.getElementById('aspectRatioWidth').type = "text";
		document.getElementById('resizeButton').style.display = 'block';
		document.getElementById('explainResize').style.display = 'block';
	}
}

function getPreProcParameter() {
	var parameters = [];
	if (document.getElementById('resizeEnableBut').checked){
		parameters.length = 3;
		parameters[0] = document.getElementById('dataDirectory').value;
		parameters[1] = document.getElementById('aspectRatioHeight').value;
		parameters[2] = document.getElementById('aspectRatioWidth').value;
	}
	else if ((document.getElementById('resizeEnableBut').checked)==false) {
		parameters.length = 1;
		parameters[0] = document.getElementById('dataDirectory').value;
	}
	return parameters;
}

// =========================================
// 	  Data Preparation - Python Functions			
// =========================================

function py_resize() {
	const argument = ['resize'];
	var parameters = getPreProcParameter();
	eel.preProcSteps(argument, parameters)
}

function py_align() {
	const argument = ['align'];
	var parameters = getPreProcParameter();
	eel.preProcSteps(argument, parameters)();
}

function py_crop() {
	const argument = ['crop'];
	var parameters = getPreProcParameter();
	eel.preProcSteps(argument, parameters)();
}

function py_augment() {
	const argument = ['augment'];
	var parameters = getPreProcParameter();
	eel.preProcSteps(argument, parameters)();
}

function py_convert() {
	const argument = ['convert'];
	var parameters = getPreProcParameter();
	eel.preProcSteps(argument, parameters)();
}

function py_training() {
	const argument = ['training'];
	var parameters = getPreProcParameter();
	eel.preProcSteps(argument, parameters)();
}

function py_segment() {
	const argument = ['segment'];
	var parameters = getPreProcParameter();
	eel.preProcSteps(argument, parameters)();
}