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

[Types]
Name: "full"; Description: "Plugins + App"; Flags: iscustom
Name: "plugins"; Description: "VST3 Plugin only"
Name: "app"; Description: "Standalone App only"

[Components]
Name: "plugins"; Description: "VST3 Plugin (for DAW: Ableton, Reaper, FL Studio, etc)"; Types: full plugins; Flags: fixed
Name: "app"; Description: "Standalone App (scan music, train, generate)"; Types: full app

[Files]
; VST3 plugin
Source: "..\..\vst\build-windows\Release\0MGE.vst3\*"; DestDir: "{commoncf64}\VST3\0MGE.vst3"; Flags: recursesubdirs ignoreversion; Components: plugins
Source: "..\..\vst\build\0MGE.vst3\*"; DestDir: "{commoncf64}\VST3\0MGE.vst3"; Flags: recursesubdirs ignoreversion skipifsourcedoesntexist; Components: plugins

; Standalone app
Source: "..\..\vst\build-windows\Release\0MGE.exe"; DestDir: "{app}"; Flags: ignoreversion; Components: app
Source: "..\..\vst\build\Release\0MGE.exe"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist; Components: app

[Icons]
Name: "{group}\Uninstall 0MGE"; Filename: "{uninstallexe}"

[Run]
Filename: "{cmd}"; Parameters: "/C echo Restart your DAW to see 0MGE plugin."; Flags: runhidden
