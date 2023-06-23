package com.google.android.exoplayer2.source.hls.offline;

public final class MasterPlaylistTestData {

  private MasterPlaylistTestData() {}

  public static final String URI = "test.m3u8";

  public static final String MEDIA_PLAYLIST_1_URI = "gear1/prog_index.m3u8";

  public static final String MEDIA_PLAYLIST_2_URI = "gear2/prog_index.m3u8";

  public static final String MEDIA_PLAYLIST_3_URI = "gear3/prog_index.m3u8";

  public static final String MEDIA_PLAYLIST_0_URI = "gear0/prog_index.m3u8";

  public static final String DATA =
      "#EXTM3U\n"
          + "#EXT-X-STREAM-INF:BANDWIDTH=232370,CODECS=\"mp4a.40.2, avc1.4d4015\"\n"
          + MEDIA_PLAYLIST_1_URI
          + "\n"
          + "#EXT-X-STREAM-INF:BANDWIDTH=649879,CODECS=\"mp4a.40.2, avc1.4d401e\"\n"
          + MEDIA_PLAYLIST_2_URI
          + "\n"
          + "#EXT-X-STREAM-INF:BANDWIDTH=991714,CODECS=\"mp4a.40.2, avc1.4d401e\"\n"
          + MEDIA_PLAYLIST_3_URI
          + "\n"
          + "#EXT-X-STREAM-INF:BANDWIDTH=41457,CODECS=\"mp4a.40.2\"\n"
          + MEDIA_PLAYLIST_0_URI;
}