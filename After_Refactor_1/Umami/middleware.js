import { NextResponse } from 'next/server';

const COLLECT_ENDPOINT = process.env.COLLECT_API_ENDPOINT;
const TRACKER_SCRIPT_NAME = process.env.TRACKER_SCRIPT_NAME;

export const config = {
  matcher: '/:path*',
};

function rewriteUrl(req, pathname, endpoint, newPathname) {
  const url = req.nextUrl.clone();

  if (pathname.endsWith(endpoint)) {
    url.pathname = newPathname;
    return NextResponse.rewrite(url);
  }
}

function customCollectEndpoint(req) {
  return rewriteUrl(req, req.nextUrl.pathname, COLLECT_ENDPOINT, '/api/send');
}

function customScriptName(req) {
  const names = TRACKER_SCRIPT_NAME.split(',').map(name => name.trim().replace(/^\/+/, ''));
  return rewriteUrl(req, req.nextUrl.pathname, names.find(name => pathname.endsWith(name)), '/script.js');
}

export default function middleware(req) {
  const fns = [customCollectEndpoint, customScriptName];

  for (const fn of fns) {
    const res = fn(req);
    if (res) {
      return res;
    }
  }

  return NextResponse.next();
}

