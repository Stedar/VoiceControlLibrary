using System;
using System.Runtime.InteropServices;


[ComVisible(true)]
[Guid("01B53C82-E921-4000-83CD-1E5D94D41D55")]
[InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
public interface IVoiceManager
{
    bool IsRecordStarted();
}
