using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace AutoTracker
{
	class USMCDLL
	{
		const string dllname = "USMCDLL.dll";

		#region Structures
		// Structure representing connected devices
		[StructLayout(LayoutKind.Sequential)]
		public unsafe struct USMC_Devices
		{
			public uint NOD;			// Number of the devices ready to work
			public char** Serial;		// Array of 16 byte ASCII strings
			public char **Version;		// Array of 4 byte ASCII strings
		};			// Structure representing connected devices
		
		// Structure representing divice state
		[StructLayout(LayoutKind.Sequential)]
		public struct State {
			public int CurPos;			// Current position (in microsteps)
			public float Temp;			// Current temperature of the driver
			public byte SDivisor;		// Step is divided by this factor
			public int bLoft;			// Indicates backlash status
			public int bFullPower;		// If TRUE then full power.
			public int bCW_CCW;		// Current direction. Relatively!
			public int bPower;			// If TRUE then Step Motor is ON.
			public int bFullSpeed;		// If TRUE then full speed. Valid in "Slow Start" mode only.
			public int bAReset;		// TRUE After Device reset, FALSE after "Set Position".
			public int bRUN;			// Indicates if step motor is rotating
			public int bSyncIN;		// Logical state directly from input synchronization PIN
			public int bSyncOUT;		// Logical state directly from output synchronization PIN
			public int bRotTr;			// Indicates current rotary transducer press state
			public int bRotTrErr;		// Indicates rotary transducer error flag
			public int bEmReset;		// Indicates state of emergency disable button (local control)
			public int bTrailer1;		// Indicates trailer 1 logical state.
			public int bTrailer2;		// Indicates trailer 2 logical state.
			public float Voltage;		// Input power source voltage (6-39V) -=24 version 0nly=-

			[MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
			byte[] Reserved;	// <Unused> File padding
		} 

		[StructLayout(LayoutKind.Sequential)]
		public struct USMC_Parameters
		{
			public float AccelT;		// Acceleration time (in ms)
			public float DecelT;		// Deceleration time (in ms)
			public float PTimeout;		// Time (in ms) after which current will be reduced to 60% of normal
			public float BTimeout1;	// Time (in ms) after which speed of step motor rotation will be equal to the one specified at
								// BTO1P field (see below). (This parameter is used when controlling step motor using buttons)
			public float BTimeout2;	//
			public float BTimeout3;	//
			public float BTimeout4;	//
			public float BTimeoutR;	// Time (in ms) after which reset command will be performed
			public float BTimeoutD;	// This field is reserved for future use
			public float MinP;			// Speed (steps/sec) while performing reset operation. (This parameter is used when controlling
								// step motor using buttons)
			public float BTO1P;		// Speed (steps/sec) after BTIMEOUT 1 time have passed. (This parameter is used when controlling
								// step motor using buttons)
			public float BTO2P;		//
			public float BTO3P;		//
			public float BTO4P;		//
			public ushort MaxLoft;		// Value in full steps that will be used performing backlash operation
			public uint StartPos;		// Current Position Saved to FLASH (see Test MicroSMC.cpp)
			public ushort RTDelta;		// Revolution distance – number of full steps per one full revolution
			public ushort RTMinError;	// Number of full steps missed to raise the error flag
			public float MaxTemp;		// Maximum allowed temperature (Celsius)
			public byte SynOUTP;		// Duration of the output synchronization pulse
			public float LoftPeriod;	// Speed of the last phase of the backlash operation.
			public float EncMult;		// Should be <Encoder Steps per Evolution> / <SM Steps per Evolution> and should be integer multiplied by 0.25

			[MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
			public byte[] Reserved;	// <Unused> File padding
		}

		// Structure representing start function parameters
		[StructLayout(LayoutKind.Sequential)]
		public struct USMC_StartParameters
		{
			public byte SDivisor;		// Step is divided by this factor (1,2,4,8)
			public int bDefDir;		// Direction for backlash operation (relative)
			public int bLoftEn;		// Enable automatic backlash operation (works if slow start/stop mode is off)
			public int bSlStart;		// If TRUE slow start/stop mode enabled.
			public int bWSyncIN;		// If TRUE controller will wait for input synchronization signal to start
			public int bSyncOUTR;		// If TRUE output synchronization counter will be reset
			public int bForceLoft;		// If TRUE and destination position is equal to the current position backlash operation will be performed.
			public uint reserved;	// <Unused> File padding
		}

		// Structure representing some of divice parameters
		[StructLayout(LayoutKind.Sequential)]
		public struct USMC_Mode
		{
			public int bPMode;			// Turn off buttons (TRUE - buttons disabled)
			public int bPReg;			// Current reduction regime (TRUE - regime is on)
			public int bResetD;		// Turn power off and make a whole step (TRUE - apply)
			public int bEMReset;		// Quick power off
			public int bTr1T;			// Trailer 1 TRUE state (TRUE : +3/+5Â; FALSE : 0Â)
			public int bTr2T;			// Trailer 2 TRUE state (TRUE : +3/+5Â; FALSE : 0Â)
			public int bRotTrT;		// Rotary Transducer TRUE state (TRUE : +3/+5Â; FALSE : 0Â)
			public int bTrSwap;		// If TRUE, Trailers are treated to be swapped
			public int bTr1En;			// If TRUE Trailer 1 Operation Enabled
			public int bTr2En;			// If TRUE Trailer 2 Operation Enabled
			public int bRotTeEn;		// If TRUE Rotary Transducer Operation Enabled
			public int bRotTrOp;		// Rotary Transducer Operation Select (stop on error for TRUE)
			public int bButt1T;		// Button 1 TRUE state (TRUE : +3/+5Â; FALSE : 0Â)
			public int bButt2T;		// Button 2 TRUE state (TRUE : +3/+5Â; FALSE : 0Â)
			public int bResetRT;		// Reset Rotary Transducer Check Positions (need one full revolution before it can detect error)
			public int bSyncOUTEn;		// If TRUE output syncronization enabled
			public int bSyncOUTR;		// If TRUE output synchronization counter will be reset
			public int bSyncINOp;		// Synchronization input mode:
								// True - Step motor will move one time to the destination position
								// False - Step motor will move multiple times by steps equal to the value destination position
			public ushort SyncCount;	// Number of steps after which synchronization output sygnal occures
			public int bSyncInvert;	// Set to TRUE to invert output synchronization signal

			public int bEncoderEn;		// Enable Encoder on pins {SYNCIN,ROTTR} - disables Synchronization input and Rotary Transducer
			public int bEncoderInv;	// Invert Encoder Counter Direction
			public int bResBEnc;		// Reset <Encoder Position> and <SM Position in Encoder units> to 0
			public int bResEnc;		// Reset <SM Position in Encoder units> to <Encoder Position>

			[MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
			public byte[] Reserved;	// <Unused> File padding
		}

		// Structure representing divice state
		[StructLayout(LayoutKind.Sequential)]
		public struct USMC_State
		{
			public int CurPos;			// Current position (in microsteps)
			public float Temp;			// Current temperature of the driver
			public byte SDivisor;		// Step is divided by this factor
			public int bLoft;			// Indicates backlash status
			public int bFullPower;		// If TRUE then full power.
			public int bCW_CCW;		// Current direction. Relatively!
			public int bPower;			// If TRUE then Step Motor is ON.
			public int bFullSpeed;		// If TRUE then full speed. Valid in "Slow Start" mode only.
			public int bAReset;		// TRUE After Device reset, FALSE after "Set Position".
			public int bRUN;			// Indicates if step motor is rotating
			public int bSyncIN;		// Logical state directly from input synchronization PIN
			public int bSyncOUT;		// Logical state directly from output synchronization PIN
			public int bRotTr;			// Indicates current rotary transducer press state
			public int bRotTrErr;		// Indicates rotary transducer error flag
			public int bEmReset;		// Indicates state of emergency disable button (local control)
			public int bTrailer1;		// Indicates trailer 1 logical state.
			public int bTrailer2;		// Indicates trailer 2 logical state.
			public float Voltage;		// Input power source voltage (6-39V) -=24 version 0nly=-

			[MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
			public byte[] Reserved;	// <Unused> File padding
		}

		// New For Firmware Version 2.4.1.0 (0x2410)
		[StructLayout(LayoutKind.Sequential)]
		public struct USMC_EncoderState
		{
			public int EncoderPos;		// Current position measured by encoder
			public int ECurPos;		// Current position (in Encoder Steps) - Synchronized with request call

			[MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
			public byte[] Reserved;	// <Unused> File padding
		}

		// ~New For Firmware Version 2.4.1.0 (0x2410)

		[StructLayout(LayoutKind.Sequential)]
		public struct USMC_Info
		{
			[MarshalAs(UnmanagedType.ByValTStr, SizeConst=17)]
			public string serial;
			public uint dwVersion;    
			[MarshalAs(UnmanagedType.ByValTStr, SizeConst=32)]
			public char DevName;
			public int CurPos, DestPos;
			public float Speed;
			public int bErrState;

			[MarshalAs(UnmanagedType.ByValArray, SizeConst=16)]
			public byte[] Reserved;	// <Unused> File padding
		}

		#endregion

		[DllImport(dllname, EntryPoint="USMC_Init", CallingConvention=CallingConvention.Cdecl)] public static extern uint _USMC_Init(out USMC_Devices devices);		// OUT - Array of structures describing all devices (may be NULL) MUST NOT be deleted

		public struct USMC_Device {
			public string serial, version;
		}

		public unsafe static USMC_Device[] USMC_Init()
		{
			USMC_Devices d;
			if (_USMC_Init(out d) != 0)
					throw new ApplicationException("Unable to initialize stepper motor library");
			USMC_Device[] devices = new USMC_Device[d.NOD];

			for (int i=0;i<d.NOD;i++) {
				devices[i] = new USMC_Device() { serial = new string(d.Serial[i], 0,16), version = new string(d.Version[i],0, 4) };
			}
			return devices;
		}


		//
		//	The USMC_GetState function returns structure representing current state of device
		//
		[DllImport(dllname, CallingConvention = CallingConvention.Cdecl)]
		public static extern uint USMC_GetState(uint Device, out State state);

		
		//
		//	The USMC_SaveParametersToFlash function saves current parameters of controller in static memory
		//	so thay can be loaded at start up time
		//
		[DllImport(dllname)] public static extern uint USMC_SaveParametersToFlash( uint Device );
		//
		//	The USMC_SetCurrentPosition function sets current position of controller
		//
		[DllImport(dllname)] public static extern uint USMC_SetCurrentPosition( uint Device,
											   int Position		// IN - New position
											   );
		//
		//	The USMC_GetMode function returns USMC_Mode structure last sent to device
		//
		[DllImport(dllname)] public static extern uint USMC_GetMode( uint Device,
										out USMC_Mode mode				// OUT - Structure representing some of device parameters
										);
		//
		//	The USMC_SetMode function sets some of device parameters
		//
		[DllImport(dllname)] public static extern uint USMC_SetMode( uint Device,
										ref USMC_Mode mode // IN/OUT Structure representing some of device parameters
										);
		//
		//	The USMC_GetParameters function returns USMC_Parameters structure last sent to device
		//
		[DllImport(dllname)] public static extern uint USMC_GetParameters( uint Device, out USMC_Parameters param	// OUT - Structure representing some of device parameters
												);
		//
		//	The USMC_SetParameters function sets some of divice parameters
		//
		[DllImport(dllname)] public static extern uint USMC_SetParameters( uint Device, ref USMC_Parameters param	// IN/OUT Structure representing some of device parameters
												);
		//
		//	The USMC_GetStartParameters function returns USMC_StartParameters structure last sent to device
		//
		[DllImport(dllname)] public static extern uint USMC_GetStartParameters( uint Device, out USMC_StartParameters startParams	// OUT - Structure representing start function parameters
													);
		//
		//	The USMC_Start function sets start parameters and starts motion
		//
		[DllImport(dllname)] public static extern uint USMC_Start( uint Device,	
										int DestPos,					// IN - Destination position
										ref float speed,					// IN/OUT - Speed of rotation
										ref USMC_StartParameters startParam		// IN/OUT - Structure representing start function parameters
										);
		//
		//	The USMC_Stop function stops device
		//
		[DllImport(dllname)] public static extern uint USMC_Stop( uint Device );
		//
		//	The USMC_GetLastErr function return string representing last error
		//
		[DllImport(dllname)] public static extern void USMC_GetLastErr(StringBuilder str,				// OUT - String buffer
											uint len				// IN - Lenght of that string buffer in bytes
											);

		//
		//	The USMC_GetDllVersion function returnes version values of USMCDLL.dll
		//
		[DllImport(dllname)] public static extern void USMC_GetDllVersion( out uint dwHighVersion,	// OUT - High Version Value
												out uint dwLowVersion);	// OUT - Low Version Value

		//
		//	The USMC_Close function closes virtual driver window "microsmc.exe"
		//
		[DllImport(dllname)] public static extern uint USMC_Close();

		//
		//	The USMC_RestoreCurPos function checks AReset bit and if it is TRUE
		//  restores previous CurPos value
		//
		[DllImport(dllname)] public static extern uint USMC_RestoreCurPos(uint device);

		// New For Firmware Version 2.4.1.0 (0x2410)

		//
		//	The USMC_GetEncoderState function returns structure representing current position of encoder
		//
		[DllImport(dllname)] public static extern uint USMC_GetEncoderState( uint Device,			// IN - Device number
												ref USMC_EncoderState encoderState	// IN/OUT Structure containing encoder state
												);

    }


}
