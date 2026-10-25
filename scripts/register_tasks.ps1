# Registers the SBTrade scheduled tasks for the current user (no admin needed). Safe to re-run.
# Times are local (this machine is on Eastern time). Tasks run only while you are logged in.
# A task with Repeat set runs again every that many minutes for RepeatMinutes after its start time.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$user = "$env:USERDOMAIN\$env:USERNAME"
$weekdays = "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"

$tasks = @(
    @{ Name = "SBTrade-swing-scan"; Job = "scan";    At = "09:00"; Note = "S&P 500 + holdings + watchlist swing scan, before the open (uses the prior completed bar)" },
    @{ Name = "SBTrade-longterm";   Job = "longterm"; At = "09:25"; Note = "Long-term stock recommendations, after the scan refreshes price history" },
    @{ Name = "SBTrade-quick-refresh"; Job = "scan_light"; At = "09:35"; Note = "Holdings + watchlist swing refresh right after the open" },
    @{ Name = "SBTrade-cc-monitor"; Job = "ccmonitor"; At = "09:35"; Repeat = 15; RepeatMinutes = 380; Note = "Covered-call monitor every 15 minutes during market hours (snapshots kept 7 days)" },
    @{ Name = "SBTrade-buys"; Job = "buys"; At = "09:40"; Repeat = 15; RepeatMinutes = 375; Note = "Buy ideas (lottos, swing calls/puts, MAG7, 0DTE) every 15 minutes during market hours" },
    @{ Name = "SBTrade-whales"; Job = "whales"; At = "09:32"; Repeat = 15; RepeatMinutes = 385; Note = "Whale activity (large option premium, fresh volume) every 15 minutes during market hours" },
    @{ Name = "SBTrade-options";    Job = "options"; At = "10:00"; Note = "Naked calls, cash-secured puts, naked puts, wheel" }
)

foreach ($t in $tasks) {
    $action = New-ScheduledTaskAction -Execute $python -Argument "-m tool.cli run-job $($t.Job)" -WorkingDirectory $root
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At $t.At
    if ($t.Repeat) {
        $repeat = New-ScheduledTaskTrigger -Once -At $t.At -RepetitionInterval (New-TimeSpan -Minutes $t.Repeat) `
            -RepetitionDuration (New-TimeSpan -Minutes $t.RepeatMinutes)
        $trigger.Repetition = $repeat.Repetition
    }
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 2)
    $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $t.Name -Action $action -Trigger $trigger -Settings $settings `
        -Principal $principal -Description $t.Note -Force | Out-Null
    $when = if ($t.Repeat) { "weekdays from $($t.At), every $($t.Repeat) min for $($t.RepeatMinutes) min" } else { "weekdays at $($t.At)" }
    Write-Output "Registered $($t.Name): $when"
}

# Schwab's refresh token dies after 7 days and the sign-in needs you at the browser, so it cannot run unattended.
# This opens the Schwab sign-in page every Monday at 08:45, before the 09:00 scan; sign in once and approve.
$login = New-ScheduledTaskAction -Execute $python -Argument "-m tool.cli schwab-login" -WorkingDirectory $root
$loginTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At "08:45"
$loginSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
$loginPrincipal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName "SBTrade-schwab-login" -Action $login -Trigger $loginTrigger -Settings $loginSettings `
    -Principal $loginPrincipal -Description "Weekly Schwab sign-in (opens the browser; you approve)" -Force | Out-Null
Write-Output "Registered SBTrade-schwab-login: Mondays at 08:45 (needs you to sign in)"
