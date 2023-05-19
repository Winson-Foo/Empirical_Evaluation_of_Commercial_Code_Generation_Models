import { NextResponse } from 'next/server';

// This function checks if the given pathname ends with the provided endpoint.
function endsWithEndpoint(req, endpoint) {
  const url = req.nextUrl.clone();
  const { pathname } = url;
  if (pathname.endsWith(endpoint)) {
    url.pathname = '/api/send';
    return NextResponse.rewrite(url);
  }
}

// This function checks if the given pathname ends with any of the provided script names.
function endsWithScriptName(req, scriptNames) {
  const url = req.nextUrl.clone();
  const { pathname } = url;
  const names = scriptNames.split(',').map(name => name.trim().replace(/^\/+/, ''));
  if (names.find(name => pathname.endsWith(name))) {
    url.pathname = '/script.js';
    return NextResponse.rewrite(url);
  }
}

// This middleware function checks if the requested URL matches any of the custom endpoints.
export default function customMiddleware(req) {
  const collectEndpoint = process.env.COLLECT_API_ENDPOINT;
  const scriptNames = process.env.TRACKER_SCRIPT_NAME;
  if (collectEndpoint) {
    const res = endsWithEndpoint(req, collectEndpoint);
    if (res) {
      return res;
    }
  }
  if (scriptNames) {
    const res = endsWithScriptName(req, scriptNames);
    if (res) {
      return res;
    }
  }
  return NextResponse.next();
}

// This config object specifies the URL pattern to match for this middleware.
export const config = {
  matcher: '/:path*',
};

