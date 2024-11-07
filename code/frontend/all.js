"use strict";

// Functions usable on all pages

eel.expose(updateDirectoryName);
function updateDirectoryName(directory, elementId) { document.getElementById(elementId).value = directory; }

function py_directory(elementId) { eel.getDirectory(elementId)(); }

function py_file(elementId) { eel.getFile(elementId)(); }

function closeAllWindows() {
	eel.windowClosed();
	alert('The Tab can close now.');
}