"use strict";

const elementsInput = document.getElementsByTagName('input');
const elementsOutput = document.getElementsByTagName('output');

function py_evaluateModel(multiclass) {
	var parameters = [];
	parameters.length = elementsInput.length;
	var p = [];
	p.length = 2;
	for(let i=0; i<elementsInput.length; i++) {
		p[0] = elementsInput[i].id;
		console.log(p[0]);
		switch (elementsInput[i].type) {
			case 'checkbox':
				p[1] = elementsInput[i].checked;
				break;
			case 'number':
				p[1] = elementsInput[i].value;
				break;
			default:
				p[1] = elementsInput[i].value;
		}
		console.log(p[1]);
		parameters[i]=p.slice();
	}
	console.table(parameters);
	eel.evaluateModel(multiclass, parameters)();
}

eel.expose(outputMetrics)
function outputMetrics(metrics) {	
	console.table(metrics);
	var index = 0;
	for(let i=0; i<elementsOutput.length; i++) {
		index = metrics.indexOf(elementsOutput[i].id)
		console.log(index);
		elementsOutput[i].innerHTML = metrics[index]
	}
}

eel.expose(outputMetric)
function outputMetric(metricName, metricValue) {
	console.table([metricName, metricValue]);
	var o = document.getElementById(metricName);
	console.log(o);
	o.innerHTML = metricValue;
}