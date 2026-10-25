# Praveen Levels (thinkorswim)
# Daily + weekly classic floor pivots, prior 20-day high (breakout line), 20-day SMA.
# Same math as the morning report. Use on daily or intraday charts.

declare upper;

input showDaily = yes;
input showWeekly = yes;
input showBreakout = yes;
input showSMA = yes;

# ---- Daily pivots from the previous completed day ----
def dH = high(period = AggregationPeriod.DAY)[1];
def dL = low(period = AggregationPeriod.DAY)[1];
def dC = close(period = AggregationPeriod.DAY)[1];
def dP = (dH + dL + dC) / 3;
def dRng = dH - dL;

plot D_R2 = if showDaily then dP + dRng else Double.NaN;
plot D_R1 = if showDaily then 2 * dP - dL else Double.NaN;
plot D_P  = if showDaily then dP else Double.NaN;
plot D_S1 = if showDaily then 2 * dP - dH else Double.NaN;
plot D_S2 = if showDaily then dP - dRng else Double.NaN;

# ---- Weekly pivots from the previous completed week ----
def wH = high(period = AggregationPeriod.WEEK)[1];
def wL = low(period = AggregationPeriod.WEEK)[1];
def wC = close(period = AggregationPeriod.WEEK)[1];
def wP = (wH + wL + wC) / 3;
def wRng = wH - wL;

plot W_R2 = if showWeekly then wP + wRng else Double.NaN;
plot W_R1 = if showWeekly then 2 * wP - wL else Double.NaN;
plot W_P  = if showWeekly then wP else Double.NaN;
plot W_S1 = if showWeekly then 2 * wP - wH else Double.NaN;
plot W_S2 = if showWeekly then wP - wRng else Double.NaN;

# ---- Prior 20-day high and 20-day SMA ----
def dClose = close(period = AggregationPeriod.DAY);
def hi20 = Highest(high(period = AggregationPeriod.DAY), 20)[1];
plot High20 = if showBreakout then hi20 else Double.NaN;
plot SMA20 = if showSMA then Average(dClose, 20)[1] else Double.NaN;

# ---- Styling ----
D_R2.SetDefaultColor(Color.RED);   D_R2.SetStyle(Curve.SHORT_DASH);
D_R1.SetDefaultColor(Color.RED);   D_R1.SetStyle(Curve.SHORT_DASH);
D_P.SetDefaultColor(Color.CYAN);   D_P.SetLineWeight(2);
D_S1.SetDefaultColor(Color.GREEN); D_S1.SetStyle(Curve.SHORT_DASH);
D_S2.SetDefaultColor(Color.GREEN); D_S2.SetStyle(Curve.SHORT_DASH);

W_R2.SetDefaultColor(Color.DARK_RED);   W_R2.SetLineWeight(3);
W_R1.SetDefaultColor(Color.DARK_RED);   W_R1.SetLineWeight(3);
W_P.SetDefaultColor(Color.BLUE);        W_P.SetLineWeight(3);
W_S1.SetDefaultColor(Color.DARK_GREEN); W_S1.SetLineWeight(3);
W_S2.SetDefaultColor(Color.DARK_GREEN); W_S2.SetLineWeight(3);

High20.SetDefaultColor(Color.ORANGE); High20.SetPaintingStrategy(PaintingStrategy.POINTS);
SMA20.SetDefaultColor(Color.GRAY);

# ---- Alerts (fire inside thinkorswim; enable app notifications for phone pushes) ----
Alert(showBreakout and close crosses above hi20, "Close above 20-day high", Alert.BAR, Sound.Ding);
Alert(close crosses above wP, "Reclaimed weekly pivot", Alert.BAR, Sound.Ding);
Alert(close crosses below wP, "Lost weekly pivot", Alert.BAR, Sound.Ding);
Alert(close crosses above 2 * wP - wL, "Above weekly R1", Alert.BAR, Sound.Ding);
Alert(close crosses below 2 * wP - wH, "Below weekly S1", Alert.BAR, Sound.Ding);
