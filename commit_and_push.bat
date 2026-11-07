@echo off
REM Double-click to save everything in this folder to GitHub (add, commit, pull, push).
setlocal
cd /d "%~dp0"
set "URL=https://github.com/DawarAhmed27/DAV-Reseach-DataContamination.git"

where git >nul 2>nul
if errorlevel 1 goto :nogit

if exist ".git" goto :linked
echo First run: linking this folder to GitHub...
git init >nul
git remote add origin "%URL%"
:linked
git fetch origin >nul 2>nul

set "BR=main"
git rev-parse --verify origin/main >nul 2>nul
if not errorlevel 1 goto :branchok
git rev-parse --verify origin/master >nul 2>nul
if not errorlevel 1 set "BR=master"
:branchok

git rev-parse --verify HEAD >nul 2>nul
if not errorlevel 1 goto :hashead
git checkout -b %BR% >nul 2>nul
git rev-parse --verify origin/%BR% >nul 2>nul
if errorlevel 1 goto :hashead
git reset --mixed origin/%BR% >nul
:hashead

set "GN="
for /f "delims=" %%a in ('git config user.name') do set "GN=%%a"
if defined GN goto :named
set /p GN=Your name for commits:
git config user.name "%GN%"
set /p GE=Your GitHub email:
git config user.email "%GE%"
:named

git add -A
set "MSG="
set /p MSG=What changed? (press Enter for a default message):
if "%MSG%"=="" set "MSG=Update %date% %time%"
git commit -m "%MSG%"
git pull --rebase --autostash origin %BR%
git push -u origin %BR%
if errorlevel 1 goto :pushfail
echo.
echo Done. Your changes are on GitHub.
goto :end

:pushfail
echo.
echo Push failed. If a GitHub sign-in window appeared, sign in and run this file again.
echo If it says "rejected", run this file again (someone else pushed first).
goto :end

:nogit
echo Git is not installed. Install it from https://git-scm.com/download/win and run this file again.

:end
pause
