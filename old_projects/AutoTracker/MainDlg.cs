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
using NationalInstruments.Vision.*;

namespace AutoTracker
{
	public partial class AutoTrackerDlg : Form
	{
		ImaqSession session;
		ImaqBufferCollection buflist;

		Thread grabThread;
		bool stopGrab;
        uint bufNum;

        public class CameraConfig {
            public Point ROISize;
            public Point[] ROIs;
            public int framerate;
            public int exposureTime;
        }

        void SetROIs(CameraConfig cfg)
        {
            WriteCmd(String.Format(":d{0:X3}{1:X3}{2:X3}{3:X3}", cfg.ROIs[0].X, cfg.ROIs[0].Y,
                cfg.ROISize.X, cfg.ROISize.Y));
        }

        void WriteCmd(string cmd)
        {
            Console.WriteLine("Camera cmd: " + cmd);
            session.SerialConnection.Write(ASCIIEncoding.ASCII.GetBytes(cmd+"\r"), 200);
        }

		public AutoTrackerDlg()
		{
			InitializeComponent();

            CameraConfig cfg = new CameraConfig() {
                ROIs = new Point[] { new Point(0, 0) },
                ROISize = new Point(800, 600)
            };

			session = new ImaqSession("img0");

            SetROIs(cfg);
            buflist = session.CreateBufferCollection(40);
            session.Acquisition.Configure(buflist);

            bufNum = 0;
            session.Acquisition.AcquireCompleted += Acquisition_AcquireCompleted;
            session.Acquisition.AcquireAsync();

            session.Start();
		}

        void Acquisition_AcquireCompleted(object sender, AsyncCompletedEventArgs e)
        {
            uint extractedNumber = 0;
            ImaqBuffer buffer = session.Acquisition.Extract(bufNum, out extractedNumber);
            bufNum++;

            
        }

		
		private void timerUIUpdate_Tick(object sender, EventArgs e)
		{
			
		}

		private void AutoTrackerDlg_FormClosing(object sender, FormClosingEventArgs e)
		{
			stopGrab = true;
		}
	}
}
