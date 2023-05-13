// viewer.js

export class Viewer {
  // implementation
}

// viewerContext.js

import { createContext } from "react";
import { Viewer } from "./viewer";

const defaultViewer = new Viewer();
export const ViewerContext = createContext({ viewer: defaultViewer });

// other module
import { ViewerContext } from "./viewerContext";

// use ViewerContext as needed