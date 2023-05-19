import { createContext } from "react";

export function createViewerContext(viewer) {
  return createContext({ viewer });
}

// Usage:
import { createViewerContext } from "./viewerContext";
import { Viewer } from "./viewer";

const viewer = new Viewer();
export const ViewerContext = createViewerContext(viewer);

// In tests, we can inject a mock viewer:
import { createViewerContext } from "./viewerContext";
const viewerMock = { /* ... */ };
export const ViewerContext = createViewerContext(viewerMock);

