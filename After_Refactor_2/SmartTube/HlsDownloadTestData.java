package com.google.android.exoplayer2.source.hls.offline;

import com.google.android.exoplayer2.C;

import java.nio.charset.Charset;

/** Data for HLS downloading tests. */
public interface HlsDownloadTestData {

  /** URI of the master playlist. */
  public static final String MASTER_PLAYLIST_URI = "test.m3u8";

  /** Index of the first media playlist in the master playlist. */
  public static final int MASTER_MEDIA_PLAYLIST_1_INDEX = 0;

  /** Index of the second media playlist in the master playlist. */
  public static final int MASTER_MEDIA_PLAYLIST_2_INDEX = 1;

  /** Index of the third media playlist in the master playlist. */
  public static final int MASTER_MEDIA_PLAYLIST_3_INDEX = 2;

  /** Index of the fourth media playlist in the master playlist. */
  public static final int MASTER_MEDIA_PLAYLIST_0_INDEX = 3;

  /** Directory name of the first media playlist. */
  public static final String MEDIA_PLAYLIST_0_DIR = "gear0/";

  /** URI of the first media playlist relative to the master playlist. */
  public static final String MEDIA_PLAYLIST_0_URI = MEDIA_PLAYLIST_0_DIR + "prog_index.m3u8";

  /** Directory name of the second media playlist. */
  public static final String MEDIA_PLAYLIST_1_DIR = "gear1/";

  /** URI of the second media playlist relative to the master playlist. */
  public static final String MEDIA_PLAYLIST_1_URI = MEDIA_PLAYLIST_1_DIR + "prog_index.m3u8";

  /** Directory name of the third media playlist. */
  public static final String MEDIA_PLAYLIST_2_DIR = "gear2/";

  /** URI of the third media playlist relative to the master playlist. */
  public static final String MEDIA_PLAYLIST_2_URI = MEDIA_PLAYLIST_2_DIR + "prog_index.m3u8";

  /** Directory name of the fourth media playlist. */
  public static final String MEDIA_PLAYLIST_3_DIR = "gear3/";

  /** URI of the fourth media playlist relative to the master playlist. */
  public static final String MEDIA_PLAYLIST_3_URI = MEDIA_PLAYLIST_3_DIR + "prog_index.m3u8";

  /** Master playlist data. */
  public static final byte[] MASTER_PLAYLIST_DATA = buildMasterPlaylistData();

  /** Media playlist data. */
  public static final byte[] MEDIA_PLAYLIST_DATA = buildMediaPlaylistData();

  /** URI of the encrypted media playlist. */
  public static final String ENC_MEDIA_PLAYLIST_URI = "enc_index.m3u8";

  /** Encrypted media playlist data. */
  public static final byte[] ENC_MEDIA_PLAYLIST_DATA = buildEncMediaPlaylistData();

  /**
   * Builds the master playlist data.
   */
  private static byte[] buildMasterPlaylistData() {
    StringBuilder builder = new StringBuilder();
    builder.append("#EXTM3U\n");
    builder.append("#EXT-X-STREAM-INF:BANDWIDTH=232370,CODECS=\"mp4a.40.2, avc1.4d4015\"\n");
    builder.append(MEDIA_PLAYLIST_1_URI);
    builder.append("\n");
    builder.append("#EXT-X-STREAM-INF:BANDWIDTH=649879,CODECS=\"mp4a.40.2, avc1.4d401e\"\n");
    builder.append(MEDIA_PLAYLIST_2_URI);
    builder.append("\n");
    builder.append("#EXT-X-STREAM-INF:BANDWIDTH=991714,CODECS=\"mp4a.40.2, avc1.4d401e\"\n");
    builder.append(MEDIA_PLAYLIST_3_URI);
    builder.append("\n");
    builder.append("#EXT-X-STREAM-INF:BANDWIDTH=41457,CODECS=\"mp4a.40.2\"\n");
    builder.append(MEDIA_PLAYLIST_0_URI);
    return builder.toString().getBytes(Charset.forName(C.UTF8_NAME));
  }

  /**
   * Builds the media playlist data.
   */
  private static byte[] buildMediaPlaylistData() {
    StringBuilder builder = new StringBuilder();
    builder.append("#EXTM3U\n");
    builder.append("#EXT-X-TARGETDURATION:10\n");
    builder.append("#EXT-X-VERSION:3\n");
    builder.append("#EXT-X-MEDIA-SEQUENCE:0\n");
    builder.append("#EXT-X-PLAYLIST-TYPE:VOD\n");
    builder.append("#EXTINF:9.97667,\n");
    builder.append("fileSequence0.ts\n");
    builder.append("#EXTINF:9.97667,\n");
    builder.append("fileSequence1.ts\n");
    builder.append("#EXTINF:9.97667,\n");
    builder.append("fileSequence2.ts\n");
    builder.append("#EXT-X-ENDLIST");
    return builder.toString().getBytes(Charset.forName(C.UTF8_NAME));
  }

  /**
   * Builds the encrypted media playlist data.
   */
  private static byte[] buildEncMediaPlaylistData() {
    StringBuilder builder = new StringBuilder();
    builder.append("#EXTM3U\n");
    builder.append("#EXT-X-TARGETDURATION:10\n");
    builder.append("#EXT-X-VERSION:3\n");
    builder.append("#EXT-X-MEDIA-SEQUENCE:0\n");
    builder.append("#EXT-X-PLAYLIST-TYPE:VOD\n");
    builder.append("#EXT-X-KEY:METHOD=AES-128,URI=\"enc.key\"\n");
    builder.append("#EXTINF:9.97667,\n");
    builder.append("fileSequence0.ts\n");
    builder.append("#EXTINF:9.97667,\n");
    builder.append("fileSequence1.ts\n");
    builder.append("#EXT-X-KEY:METHOD=AES-128,URI=\"enc2.key\"\n");
    builder.append("#EXTINF:9.97667,\n");
    builder.append("fileSequence2.ts\n");
    builder.append("#EXT-X-ENDLIST");
    return builder.toString().getBytes(Charset.forName(C.UTF8_NAME));
  }

} 