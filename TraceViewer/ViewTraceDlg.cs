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

		class Trace
		{
			public List<Frame> frames;
			public int avg;
		}

		List<Trace> traces = new List<Trace>();

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

				ReadHeader(false);
				UpdateGraph();
			}
		}

		void ReadHeader(bool oldVersion)
		{
			traces = new List<Trace>();
			Trace tr = new Trace();
			tr.frames = new List<Frame>();
			traces.Add(tr);

			using (FileStream stream = File.OpenRead(filename))
			{
				stream.Seek(0, SeekOrigin.Begin);

				using (BinaryReader r = new BinaryReader(stream))
				{
					int a = r.ReadInt32();
					int b = r.ReadInt32();
					int c = r.ReadInt32();

					int infoCols;
					if (oldVersion)
					{
						numBeads = a;
						infoCols = b;
						startOffset = c;
					}
					else
					{
						int d = r.ReadInt32();
						infoCols = c;

						if (d == 1234)
						{
							// shitty format
							infoCols = 7;
							startOffset = (int)c;
						}
						else startOffset = (int)d;

						numBeads = b; 
						
					}
					infoColNames = new string[infoCols];
					for (int j = 0; j < infoCols; j++)
					{
						infoColNames[j] = ReadZStr(r);
						Console.WriteLine("InfoCol[{0}]={1}", j, infoColNames[j]);
					}
					haveErrors = !oldVersion;
					bytesPerFrame = 4 * 3 * numBeads + infoCols * 4 + 4 + 8 + (haveErrors ? (4 * numBeads) : 0);
					numFrames = ((int)stream.Length - startOffset) / bytesPerFrame;
					Console.WriteLine("#Frames: {0}", numFrames);

					stream.Seek(startOffset, SeekOrigin.Begin);


					for (int i = 0; i < numFrames; i++)
					{
						tr.frames.Add(ReadFrame(r));
					}

					tr.avg = 1;
					ComputeAverages();

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

		void ComputeAverages()
		{
			int i = 1;
			while (true)
			{
				Trace org = traces.LastOrDefault();
				Trace nt = new Trace();
				nt.avg = i*2;
				nt.frames = new List<Frame>(org.frames.Count / 2);

				for (int j = 0; j < org.frames.Count-1; j+=2)
				{
					Frame a=org.frames[j];
					Frame b=org.frames[j+1];

					Frame fr = new Frame();
					fr.positions = new Vec3[a.positions.Length];
					for (int k = 0; k < fr.positions.Length; k++) /*
						fr.positions[k] = new Vec3()
						{
							x = 0.5f * (a.positions[k].x + b.positions[k].x),
							y = 0.5f * (a.positions[k].y + b.positions[k].y),
							z = 0.5f * (a.positions[k].z + b.positions[k].z)
						};*/
						fr.positions[k] = a.positions[k];

					fr.frameInfo = new float[a.frameInfo.Length];
					for (int k = 0; k < a.frameInfo.Length; k++)
						fr.frameInfo[k] = a.frameInfo[k];

					nt.frames.Add(fr);
				}

				traces.Add(nt);
				i *= 2;

				if (nt.frames.Count < 1000)
					break;
			}
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

			int nf = int.Parse(textNumFramesInView.Text);

			float framesPerPixel = nf/(float)chart.Width ;
			Trace tr = traces.FirstOrDefault(t => t.avg*2>= framesPerPixel);
			if (tr == null) tr = traces[0];

			int maxFrame = tr.frames.Count;
			int nFrame = nf / tr.avg;
			int startFrame = Math.Min(trackBar.Value / tr.avg, maxFrame - nFrame);
			if (startFrame < 0) startFrame = 0;
			if (nFrame + startFrame > maxFrame) nFrame=maxFrame-startFrame;
			Console.WriteLine("StartFrame:{0} ", startFrame);
			
			Frame[] data = tr.frames.GetRange(startFrame, nFrame).ToArray();
			chart.Series.Clear();
			chart.SuspendLayout();

			int refBead = -1;
			if (checkUseRef.Checked && txtRefBead.Text.Length > 0)
			{
				refBead = int.Parse(txtRefBead.Text);
			}

			for (int i = 0; i < beads.Length; i++)
			{
				int bead=beads[i];

				Series xs = checkX.Checked ? new Series("Bead " + bead.ToString() + " X") { ChartType = SeriesChartType.FastLine } : null;
				Series ys = checkY.Checked ? new Series("Bead " + bead.ToString() + " Y") { ChartType = SeriesChartType.FastLine } : null;
				Series zs = checkZ.Checked ? new Series("Bead " + bead.ToString() + " Z") { ChartType = SeriesChartType.FastLine } : null;

				for (int j = 0; j < data.Length; j++)
				{
					var pos = data[j].positions[bead];

					if (refBead>=0) {
						Vec3 refpos=data[j].positions[refBead];
						pos.x-=refpos.x;
						pos.y-=refpos.y;
						pos.z-=refpos.z;
					}

					if (xs != null) xs.Points.AddY(pos.x);
					if (ys != null) ys.Points.AddY(pos.y);
					if (zs != null) zs.Points.AddY(pos.z);
				}

				if (xs != null) chart.Series.Add(xs);
				if (ys != null) chart.Series.Add(ys);
				if (zs != null) chart.Series.Add(zs);
			}
			if (checkMagnetZ.Checked)
			{
				Series mzs = new Series("Magnet Z") { ChartType = SeriesChartType.FastLine };
				chart.Series.Add(mzs);
				for (int i = 0; i < data.Length; i++)
					mzs.Points.AddY(data[i].frameInfo[0]);
			}
			chart.ResetAutoValues();
			chart.ResumeLayout();
			chart.Update();

			/*
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
			}*/
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

			UpdateLUT();

		}

		void UpdateLUT()
		{
			string dirpath = Path.GetDirectoryName(filename);

			string lutimg = dirpath + string.Format("\\lut\\lut{0:D3}", beadSelect.Value) + ".png";

			try
			{
				Image img = Image.FromFile(lutimg);
				lutView.Image = img;
			}
			catch (Exception e)
			{
				Console.WriteLine("Failed to load {0}. Exception: {1}", lutimg, e.Message);
			}
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

		private void openOldVersionFileToolStripMenuItem_Click(object sender, EventArgs e)
		{

			OpenFileDialog ofd = new OpenFileDialog();
			if (ofd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
			{
				filename = ofd.FileName;

				ReadHeader(true);
				UpdateGraph();
			}
		}

		private void exportZTraces_Click(object sender, EventArgs e)
		{
			SaveFileDialog sfd = new SaveFileDialog();
			if (sfd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
			{
				using (var f=sfd.OpenFile()) {
					using (StreamWriter w = new StreamWriter(f))
					{
						for (int i = 0; i < traces[0].frames.Count; i++)
						{
							Frame fr = traces[0].frames[i];
							w.Write("{0}\t", fr.timestamp);

							for (int j = 0; j < fr.positions.Length; j++)
							{
								w.Write(fr.positions[j].z.ToString());
								if (j < fr.positions.Length - 1) w.Write("\t");
							}

							w.WriteLine();
						}
					}
				}
			}
		}

		private void checkUseRef_CheckedChanged(object sender, EventArgs e)
		{
			UpdateGraph();
		}

		private void textNumFramesInView_TextChanged(object sender, EventArgs e)
		{
			UpdateGraph();
		}

		private void txtRefBead_TextChanged(object sender, EventArgs e)
		{
			UpdateGraph();
		}

		private void checkX_TextChanged(object sender, EventArgs e)
		{
			UpdateGraph();
		}

		private void checkY_CheckedChanged(object sender, EventArgs e)
		{
			UpdateGraph();

		}

		private void checkZ_CheckedChanged(object sender, EventArgs e)
		{
			UpdateGraph();

		}

		private void checkMagnetZ_CheckedChanged(object sender, EventArgs e)
		{
			UpdateGraph();
		}
			
	}
}
