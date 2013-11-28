using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;

using System.Threading;

using NationalInstruments.Vision.Acquisition.Imaq;
using NationalInstruments.Vision.Analysis.Internal;

namespace AutoTracker
{
	public partial class AutoTrackerDlg : Form
	{
		ImaqSession session;
		ImaqBufferCollection buflist;

		Thread grabThread;
		bool stopGrab;

		public AutoTrackerDlg()
		{
			InitializeComponent();

			session = new ImaqSession("img0");

			buflist = session.CreateBufferCollection(40);
			session.RingSetup(buflist, 0, false);

			grabThread = new Thread(new ThreadStart(GrabbingThread));

			session.Start();
		}

		void GrabbingThread()
		{
			uint bufferIndex;
			uint i = 0;

			while (!stopGrab)
			{
				ImaqBuffer buffer = session.Acquisition.Extract(i, out bufferIndex);


				
				i++;
			}
			session.Stop();
		}

		
		private void timerUIUpdate_Tick(object sender, EventArgs e)
		{
			
		}

		private void AutoTrackerDlg_FormClosing(object sender, FormClosingEventArgs e)
		{
			stopGrab = true;
			while (grabThread.ThreadState == ThreadState.Running)
				;
		}
	}
}
