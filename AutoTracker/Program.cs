using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;

namespace AutoTracker
{
	static class Program
	{
		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main()
		{
			var d = USMCDLL.USMC_Init();
			for (uint i=0;i<d.Length;i++)
			{
				var dev = d[i];
//				Console.WriteLine("Device serial: {0}, version: {1}", dev.serial, dev.version);

				USMCDLL.State state;
				USMCDLL.USMC_GetState(i, out state);

				Console.WriteLine("Device {0}", i);
			}

			Application.EnableVisualStyles();
			Application.SetCompatibleTextRenderingDefault(false);
			Application.Run(new AutoTrackerDlg() );
		}
	}
}
