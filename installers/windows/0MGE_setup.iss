; 0MGE Windows Installer - Inno Setup Script
; Download Inno Setup: https://jrsoftware.org/isinfo.php

[Setup]
AppId={{0MGE-0PENAGI-GRANULAR-ENGINE}
AppName=0MGE
AppVersion=1.0.0
AppPublisher=0penAGI
DefaultDirName={commoncf}\0MGE
DefaultGroupName=0MGE
OutputDir=Output
OutputBaseFilename=0MGE-1.0.0-Windows-Setup
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=admin

[Files]
; VST3 plugin - works with both local build and CI build
Source: "..\..\vst\build-windows\Release\0MGE.vst3\*"; DestDir: "{commoncf64}\VST3\0MGE.vst3"; Flags: recursesubdirs ignoreversion
Source: "..\..\vst\build\0MGE.vst3\*"; DestDir: "{commoncf64}\VST3\0MGE.vst3"; Flags: recursesubdirs ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\Uninstall 0MGE"; Filename: "{uninstallexe}"

[Run]
Filename: "{cmd}"; Parameters: "/C echo Restart your DAW to see 0MGE plugin."; Flags: runhidden
