package com.facebook.perftest;

public interface IPerfTestConfig {
  boolean isRunningInPerfTest();
}

public class PerfTestConfig implements IPerfTestConfig {
  @Override
  public boolean isRunningInPerfTest() {
    return false;
  }
} 