# Overnight pre-registration jobs (PILOT_LOG 15.20). Nothing here touches test subjects or confirmatory units.
# Run from the inner repo folder:  Set-ExecutionPolicy -Scope Process Bypass; .\overnight_20260923.ps1
$ErrorActionPreference = 'Continue'
$py   = 'D:\miniconda\python.exe'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$log  = Join-Path $root 'logs\overnight_20260923.log'
function Step($name) { $t = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'; "`n===== [$t] $name =====" | Tee-Object -FilePath $log -Append }
function Run { param([string[]]$a) & $py @a 2>&1 | Tee-Object -FilePath $log -Append; "exit code: $LASTEXITCODE" | Tee-Object -FilePath $log -Append }

Step 'pytest'
Run @('-m','pytest','-q')

Step 'P9 (watchdog)'
Run @('watchdog.py','--watch','results/pilots','--','pilot','P9')

Step 'diag_cbramod arm C'
Run @('diag_cbramod.py','--arms','C','--epochs','8')

# ---------------- SHU-MI (figshare 19228725, mat_files.zip, MD5 6c039cce4025b2749545949c93f7a4f1)
Step 'SHU-MI download'
$shu = Join-Path $HOME 'data\SHU-MI'
New-Item -ItemType Directory -Force $shu | Out-Null
$zip = Join-Path $shu 'mat_files.zip'
if (-not (Test-Path $zip)) { curl.exe -sS -L --retry 5 -o $zip https://ndownloader.figshare.com/files/36728994 2>&1 | Tee-Object -FilePath $log -Append }
$md5 = (Get-FileHash $zip -Algorithm MD5).Hash
"MD5 $md5" | Tee-Object -FilePath $log -Append
if ($md5 -eq '6C039CCE4025B2749545949C93F7A4F1') {
  if (-not (Get-ChildItem $shu -Recurse -Filter *.mat -ErrorAction SilentlyContinue | Select-Object -First 1)) {
    Run @('unzip_any.py', $zip, $shu) }   # Expand-Archive fails: entries use Deflate64 (PILOT_LOG 15.21)
  Step 'P11 (watchdog)'
  Run @('watchdog.py','--watch','results/pilots','--','pilot','P11')
} else { 'MD5 MISMATCH - P11 skipped' | Tee-Object -FilePath $log -Append }

# ---------------- EEGEyeNet (OSF ktv7m, Direction_task_with_dots_synchronised_min.npz)
Step 'EEGEyeNet download'
function Find-OsfFile($url, $name) {
  while ($url) {
    $r = Invoke-RestMethod $url
    foreach ($e in $r.data) {
      if ($e.attributes.kind -eq 'file' -and $e.attributes.name -eq $name) { return $e }
      if ($e.attributes.kind -eq 'folder') {
        $f = Find-OsfFile $e.relationships.files.links.related.href $name
        if ($f) { return $f } } }
    $url = $r.links.next } }
$eye = Join-Path $HOME 'data\EEGEyeNet'
New-Item -ItemType Directory -Force $eye | Out-Null
$npz = Join-Path $eye 'Direction_task_with_dots_synchronised_min.npz'
try {
  if (-not (Test-Path $npz)) {
    $f = Find-OsfFile 'https://api.osf.io/v2/nodes/ktv7m/files/osfstorage/' 'Direction_task_with_dots_synchronised_min.npz'
    if ($f) {
      ('OSF size {0:N2} GB' -f ($f.attributes.size / 1GB)) | Tee-Object -FilePath $log -Append
      curl.exe -sS -L --retry 5 -o $npz $f.links.download 2>&1 | Tee-Object -FilePath $log -Append
    } else { 'OSF file not found via API - download manually from osf.io/ktv7m' | Tee-Object -FilePath $log -Append } }
} catch { "OSF error: $_" | Tee-Object -FilePath $log -Append }
if (Test-Path $npz) {
  Step 'P10a (watchdog)'
  Run @('watchdog.py','--watch','results/pilots','--','pilot','P10a','--npz',$npz)
}

Step 'DONE'
