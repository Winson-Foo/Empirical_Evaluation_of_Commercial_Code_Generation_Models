import { createContext } from "react";
import { Viewer } from "./viewer";

// create a new instance of the Viewer class
const myViewer = new Viewer();

// create the context with the viewer instance as the default value
export const ViewerContext = createContext({ myViewer });

// export the viewer instance separately (in case we need it elsewhere)
export const viewerInstance = myViewer;

// if we want to add functionality to the Viewer class, we can do so like this:

// create a new class that extends the Viewer class
class MyViewerClass extends Viewer {
  constructor() {
    super();
    // add any new properties or methods here
  }
}

// create a new instance of the custom Viewer class
const myCustomViewer = new MyViewerClass();

// export the custom Viewer instance separately (in case we need it elsewhere)
export const customViewerInstance = myCustomViewer;

// update the context to use the custom Viewer instance as the default value
export const CustomViewerContext = createContext({ myCustomViewer });

