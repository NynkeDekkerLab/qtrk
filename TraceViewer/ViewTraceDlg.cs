using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Windows.Forms.DataVisualization.Charting;

namespace TraceViewer
{
	public partial class ViewTraceDlg : Form, IDisposable
	{
		int numFrames;
		int numBeads;
		string[] infoColNames;
		int startOffset;
		int bytesPerFrame;
		bool haveErrors;
		private string filename;

		public ViewTraceDlg()
		{
			InitializeComponent();
		}

		private void readBinFileToolStripMenuItem_Click(object sender, EventArgs e)
		{

		}

		private void menuOpenFile(object sender, EventArgs e)
		{
			OpenFileDialog ofd = new OpenFileDialog();
			if (ofd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
			{
				filename = ofd.FileName;

				ReadHeader();
				UpdateGraph();
			}
		}

		void ReadHeader()
		{
			using (FileStream stream = File.OpenRead(filename))
			{
				stream.Seek(0, SeekOrigin.Begin);

				using (BinaryReader r = new BinaryReader(stream))
				{
					int a = r.ReadInt32();
					int b = r.ReadInt32();
					int c = r.ReadInt32();
					int d = r.ReadInt32();

					int infoCols = c;

					if (d == 1234)
					{
						// shitty format
						infoCols = 7;
					}
					numBeads = b;
					infoColNames = new string[infoCols];
					for (int j = 0; j < infoCols; j++)
					{
						infoColNames[j] = ReadZStr(r);
						Console.WriteLine("InfoCol[{0}]={1}", j, infoColNames[j]);
					}

					if (d == 1234)
					{
						while (r.ReadInt32() == 0) ;
						startOffset = (int)c;
					}
					else
					{
						startOffset = (int)d;
					}
					haveErrors = true;
					bytesPerFrame = 4 * 3 * numBeads + infoCols * 4 + 4 + 8 + (haveErrors ? (4 * numBeads) : 0);
					numFrames = ((int)stream.Length - startOffset) / bytesPerFrame;
					Console.WriteLine("#Frames: {0}", numFrames);

					stream.Seek(startOffset, SeekOrigin.Begin);

					for (int f = 0; f < 4; f++)
					{
						var fr = ReadFrame(r);

						Console.WriteLine("Frame#: {0}", fr.id);
						Console.WriteLine("Timestamp: {0}", fr.timestamp);

						for (int i = 0; i < numBeads; i++)
							Console.WriteLine("Bead[{0}]: X={1},Y={2},Z={3}", i, fr.positions[i].x, fr.positions[i].y, fr.positions[i].z);
					}

					beadSelect.Maximum = numBeads-1;
					beadSelect.Value = 0;
					trackBar.Maximum = numFrames;
					trackBar.Value = 0;
				}

			}
		}

		struct Vec3 {
			public float x,y,z;
		}

		class Frame
		{
			public int id;
			public double timestamp;
			public float[] frameInfo;
			public Vec3[] positions;
			public int[] errors;
		}

		Frame ReadFrame(BinaryReader r)
		{
			Frame f = new Frame();
			f.frameInfo = new float[infoColNames.Length];
			f.positions = new Vec3[numBeads];
			f.id = r.ReadInt32();
			f.timestamp = r.ReadDouble();
			for (int i = 0; i < infoColNames.Length; i++)
				f.frameInfo[i] = r.ReadSingle();
			for (int i = 0; i < numBeads; i++)
			{
				Vec3 v = new Vec3() {
					x=r.ReadSingle(),
					y=r.ReadSingle(),
					z=r.ReadSingle()
				};
				f.positions[i] = v;
			}
			if (haveErrors)
			{
				f.errors = new int[numBeads];
				for (int i = 0; i < numBeads; i++)
					f.errors[i] = r.ReadInt32();
			}
			return f;
		}

		string ReadZStr(BinaryReader r)
		{
			StringBuilder sb = new StringBuilder();
			while (true)
			{
				char b = r.ReadChar();
				if (b == 0)
					return sb.ToString();
				sb.Append(b);
			}
		}

		void UpdateGraph()
		{
			int[] beads = new int[] { beadSelect.Value };

			if (filename == null)
				return;

			using (FileStream s = File.OpenRead(filename))
			{
				int nf = int.Parse(textNumFramesInView.Text);
				Frame[] data = ReadFrames(s, trackBar.Value, nf);
				chart.Series.Clear();
				int refBead=-1;
				if (txtRefBead.Text.Length >0) {
					refBead=int.Parse(txtRefBead.Text);
				}
				
				for (int i = 0; i < beads.Length; i++)
				{
					var series = chart.Series.Add("Bead " + i.ToString());
					series.ChartType = SeriesChartType.Line;

					for (int j = 0; j < data.Length; j++)
					{
						var pos = data[j].positions;
						double v;
						if (refBead >= 0) v = pos[beads[i]].z - pos[refBead].z;
						else v = pos[beads[i]].z;

						series.Points.AddY(v);
					}

				}
				chart.Update();
			}
		}

		private Frame[] ReadFrames(FileStream s, int start, int count)
		{
			if (start < 0) start = 0;
			if (count + start > numFrames)
				count = numFrames - start;
			if (count == 0)
				return new Frame[0];

			s.Seek(startOffset + bytesPerFrame * start, SeekOrigin.Begin);
			Frame[] data = new Frame[count];

			using(BinaryReader r = new BinaryReader(s))
			{
				for (int i = 0; i < count; i++)
					data[i] = ReadFrame(r);
			}
			return data;
		}

		private void trackBar_ValueChanged(object sender, EventArgs e)
		{
			UpdateGraph();
		}

		private void beadSelect_Scroll(object sender, EventArgs e)
		{
			labelBead.Text = beadSelect.Value.ToString();
		}

		private void trackBar_Scroll(object sender, EventArgs e)
		{
			labelFrame.Text = trackBar.Value.ToString();
		}

		private void beadSelect_ValueChanged(object sender, EventArgs e)
		{
			UpdateGraph();
		}

		private void writeAsTextToolStripMenuItem_Click(object sender, EventArgs e)
		{
			var sfd = new SaveFileDialog();
			if (sfd.ShowDialog() == System.Windows.Forms.DialogResult.OK) {

				Stream txtfile = sfd.OpenFile();

				using (FileStream s = File.OpenRead(filename)) {
				}
			}

		}

		private void repairBinaryToolStripMenuItem_Click(object sender, EventArgs e)
		{
			FileStream s = File.Open(filename, FileMode.Open, FileAccess.ReadWrite);

			s.Seek(0, SeekOrigin.Begin);
			using (BinaryWriter w = new BinaryWriter(s))
			{
				w.Write((int)2); // version
				w.Write(numBeads);
				w.Write(infoColNames.Length);
				w.Write(startOffset);

			}
		}
	}
}
