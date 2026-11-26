; Precompressed resources avoid NSIS's uncompressed solid-block size limit.
!include "LogicLib.nsh"
!include "x64.nsh"
!include "${__FILEDIR__}\build\agent-payload-uninstall.nsh"

Var TyloAgentTar
Var TyloAgentExtractExit

!macro NSIS_HOOK_PREINSTALL
  ; Reserve the expanded payload in addition to the temporary ZIP counted by File.
  AddSize ${TYLOINFER_AGENT_PAYLOAD_KIB}
  ; Windows 10 1803+ includes bsdtar with ZIP support. Use the native executable
  ; from the 32-bit NSIS process instead of relying on the user's PATH.
  StrCpy $TyloAgentTar "$SYSDIR\tar.exe"
  ${If} ${RunningX64}
    StrCpy $TyloAgentTar "$WINDIR\Sysnative\tar.exe"
  ${EndIf}
  IfFileExists "$TyloAgentTar" +4 0
    MessageBox MB_OK|MB_ICONSTOP "TyloInfer Agent requires Windows tar.exe (Windows 10 1803 or newer)." /SD IDOK
    SetErrorLevel 1
    Abort
!macroend

!macro NSIS_HOOK_POSTINSTALL
  nsExec::ExecToLog '"$TyloAgentTar" -xf "$INSTDIR\agent-payload.zip" -C "$INSTDIR"'
  Pop $TyloAgentExtractExit
  ${If} $TyloAgentExtractExit != 0
    MessageBox MB_OK|MB_ICONSTOP "Could not extract TyloInfer Agent resources (exit $TyloAgentExtractExit). Run the installer again after closing the application." /SD IDOK
    SetErrorLevel 1
    Abort
  ${EndIf}
  Delete "$INSTDIR\agent-payload.zip"
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  !insertmacro TYLOINFER_AGENT_REMOVE_PAYLOAD
!macroend
