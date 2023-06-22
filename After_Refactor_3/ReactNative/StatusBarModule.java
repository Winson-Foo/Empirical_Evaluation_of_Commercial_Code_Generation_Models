private static final String DIMENSION_TYPE = "dimen";
   private static final int STATUS_BAR_HEIGHT_ID = R.dimen.status_bar_height;
   private static final int FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS = WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS;
   private static final int FLAG_FULLSCREEN = WindowManager.LayoutParams.FLAG_FULLSCREEN;
   private static final int SYSTEM_UI_FLAG_LIGHT_STATUS_BAR = View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR;
   
   @Override
   public void setColor(final double colorValue, final boolean animated) {
     final int color = (int) colorValue;
   
     final Activity activity = getCurrentActivity();
     if (activity == null) {
       FLog.w(
           ReactConstants.TAG,
           "StatusBarModule: Ignored status bar change, current activity is null.");
       return;
     }
   
     UiThreadUtil.runOnUiThread(
         new GuardedRunnable(getReactApplicationContext()) {
           @Override
           public void runGuarded() {
             activity
                 .getWindow()
                 .addFlags(FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS);
             if (animated) {
               int currentColor = activity.getWindow().getStatusBarColor();
               ValueAnimator colorAnimation =
                   ValueAnimator.ofObject(new ArgbEvaluator(), currentColor, color);
   
               colorAnimation.addUpdateListener(new StatusBarColorUpdateListener(activity));
               colorAnimation.setDuration(300).setStartDelay(0);
               colorAnimation.start();
             } else {
               activity.getWindow().setStatusBarColor(color);
             }
           }
         });
   }
