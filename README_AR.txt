باك تست ذهب يومي 1D لشهر أكتوبر 2026، هدف 3% وثقة >=70%.
ارفع الملفات الثلاثة إلى جذر مشروع Metals على GitHub: gold_daily_backtest.py و fast_both_engine.py و both_hourly_backtest.py.
في Railway Terminal بعد النشر:
cd /app
/app/.venv/bin/python -u gold_daily_backtest.py
النتائج: /app/backtest_gold_1d_tp3_conf70
ALL_TRADES.csv يشمل سعر الدخول والهدف والوقف والخروج.
الأفق 10 جلسات يومية. الإشارات قرب نهاية البيانات التي لا تتوافر لها 10 جلسات لاحقة لا تحسب كصفقات؛ ويظل وجودها داخل ALL_SIGNALS.csv متاحا. لا تعديل للبيانات الأصلية ولا تشغيل سيرفر.
