package com.google.android.exoplayer2.source.hls.offline;

import com.google.android.exoplayer2.C;

import java.nio.charset.Charset;

/** Data for HLS downloading tests. */
public final class HlsDownloadTestData {

  public static final String MASTER_PLAYLIST_URI = "test.m3u8";

  public static final class MediaPlaylist0 {
    public static final String DIRECTORY_NAME = "gear0/";
    public static final String URI = DIRECTORY_NAME + "prog_index.m3u8";
    public static final String DATA =
        "#EXTM3U\n"
            + "#EXT-X-TARGETDURATION:10\n"
            + "#EXT-X-VERSION:3\n"
            + "#EXT-X-MEDIA-SEQUENCE:0\n"
            + "#EXT-X-PLAYLIST-TYPE:VOD\n"
            + "#EXTINF:9.97667,\n"
            + "fileSequence0.ts\n"
            + "#EXTINF:9.97667,\n"
            + "fileSequence1.ts\n"
            + "#EXTINF:9.97667,\n"
            + "fileSequence2.ts\n"
            + "#EXT-X-ENDLIST";
  }

  public static final class MediaPlaylist1 {
    public static final String DIRECTORY_NAME = "gear1/";
    public static final String URI = DIRECTORY_NAME + "prog_index.m3u8";
    public static final String DATA =
        "#EXTM3U\n"
            + "#EXT-X-TARGETDURATION:10\n"
            + "#EXT-X-VERSION:3\n"
            + "#EXT-X-MEDIA-SEQUENCE:0\n"
            + "#EXT-X-PLAYLIST-TYPE:VOD\n"
            + "#EXTINF:9.97667,\n"
            + "fileSequence0.ts\n"
            + "#EXTINF:9.97667,\n"
            + "fileSequence1.ts\n"
            + "#EXTINF:9.97667,\n"
            + "fileSequence2.ts\n"
            + "#EXT-X-ENDLIST";
  }

  public static final class MediaPlaylist2 {
    public static final String DIRECTORY_NAME = "gear2/";
    public static final String URI = DIRECTORY_NAME + "prog_index.m3u8";
    public static final String DATA =
        "#EXTM3U\n"
            + "#EXT-X-TARGETDURATION:10\n"
            + "#EXT-X-VERSION:3\n"
            + "#EXT-X-MEDIA-SEQUENCE:0\n"
            + "#EXT-X-PLAYLIST-TYPE:VOD\n"
            + "#EXTINF:9.97667,\n"
            + "fileSequence0.ts\n"
            + "#EXTINF:9.97667,\n"
            + "fileSequence1.ts\n"
            + "#EXTINF:9.97667,\n"
            + "fileSequence2.ts\n"
            + "#EXT-X-ENDLIST";
  }

  public static final class MediaPlaylist3 {
    public static final String DIRECTORY_NAME = "gear3/";
    public static final String URI = DIRECTORY_NAME + "prog_index.m3u8";
    public static final String DATA =
        "#EXTM3U\n"
            + "#EXT-X-TARGETDURATION:10\n"
            + "#EXT-X-VERSION:3\n"
            + "#EXT-X-MEDIA-SEQUENCE:0\n"
            + "#EXT-X-PLAYLIST-TYPE:VOD\n"
            + "#EXTINF:9.97667,\n"
            + "fileSequence0.ts\n"
            + "#EXTINF:9.97667,\n"
            + "fileSequence1.ts\n"
            + "#EXTINF:9.97667,\n"
            + "fileSequence2.ts\n"
            + "#EXT-X-ENDLIST";
  }

  public static final class MasterPlaylist {
    public static final int MEDIA_PLAYLIST_1_INDEX = 0;
    public static final int MEDIA_PLAYLIST_2_INDEX = 1;
    public static final int MEDIA_PLAYLIST_3_INDEX = 2;
    public static final int MEDIA_PLAYLIST_0_INDEX = 3;
    public static final String DATA =
        "#EXTM3U\n"
            + "#EXT-X-STREAM-INF:BANDWIDTH=232370,CODECS=\"mp4a.40.2, avc1.4d4015\"\n"
            + MediaPlaylist1.URI
            + "\n"
            + "#EXT-X-STREAM-INF:BANDWIDTH=649879,CODECS=\"mp4a.40.2, avc1.4d401e\"\n"
            + MediaPlaylist2.URI
            + "\n"
            + "#EXT-X-STREAM-INF:BANDWIDTH=991714,CODECS=\"mp4a.40.2, avc1.4d401e\"\n"
            + MediaPlaylist3.URI
            + "\n"
            + "#EXT-X-STREAM-INF:BANDWIDTH=41457,CODECS=\"mp4a.40.2\"\n"
            + MediaPlaylist0.URI;
  }

  public static final String ENC_MEDIA_PLAYLIST_URI = "enc_index.m3u8";

  public static final String ENC_MEDIA_PLAYLIST_DATA =
      "#EXTM3U\n"
          + "#EXT-X-TARGETDURATION:10\n"
          + "#EXT-X-VERSION:3\n"
          + "#EXT-X-MEDIA-SEQUENCE:0\n"
          + "#EXT-X-PLAYLIST-TYPE:VOD\n"
          + "#EXT-X-KEY:METHOD=AES-128,URI=\"enc.key\"\n"
          + "#EXTINF:9.97667,\n"
          + "fileSequence0.ts\n"
          + "#EXTINF:9.97667,\n"
          + "fileSequence1.ts\n"
          + "#EXT-X-KEY:METHOD=AES-128,URI=\"enc2.key\"\n"
          + "#EXTINF:9.97667,\n"
          + "fileSequence2.ts\n"
          + "#EXT-X-ENDLIST";

  private HlsDownloadTestData() {} // Prevent instantiation.
}

