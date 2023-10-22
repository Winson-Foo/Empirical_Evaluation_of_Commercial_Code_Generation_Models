import { NextResponse } from 'next/server';

const COLLECT_ENDPOINT = process.env.COLLECT_API_ENDPOINT;
const TRACKER_SCRIPT_NAME = process.env.TRACKER_SCRIPT_NAME;

function isCollectEndpoint(pathname) {
  return COLLECT_ENDPOINT && pathname.endsWith(COLLECT_ENDPOINT);
}

function isScriptName(pathname) {
  const names = TRACKER_SCRIPT_NAME ? TRACKER_SCRIPT_NAME.split(',').map(name => name.trim().replace(/^\/+/, '')) : [];
  return names.some(name => pathname.endsWith(name));
}

function rewriteUrl(url, newPathname) {
  url.pathname = newPathname;
  return NextResponse.rewrite(url);
}

function handleCollectEndpoint(req) {
  const url = req.nextUrl.clone();
  const { pathname } = url;

  if (isCollectEndpoint(pathname)) {
    return rewriteUrl(url, '/api/send');
  }
}

function handleScriptName(req) {
  const url = req.nextUrl.clone();
  const { pathname } = url;

  if (isScriptName(pathname)) {
    return rewriteUrl(url, '/script.js');
  }
}

export default function middleware(req) {
  const handlers = [handleCollectEndpoint, handleScriptName];

  for (const handler of handlers) {
    const res = handler(req);

    if (res) {
      return res;
    }
  }

  return NextResponse.next();
}