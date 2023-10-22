import { createContext } from "react";
import { Viewer } from "./viewer";

const defaultViewer = new Viewer();
export const ViewerContext = createContext({ viewer: defaultViewer });