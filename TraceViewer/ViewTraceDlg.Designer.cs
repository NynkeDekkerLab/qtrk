namespace TraceViewer
{
	partial class ViewTraceDlg
	{
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		/// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
		protected override void Dispose(bool disposing)
		{
			if (disposing && (components != null))
			{
				components.Dispose();
			}
			base.Dispose(disposing);
		}

		#region Windows Form Designer generated code

		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea1 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
			System.Windows.Forms.DataVisualization.Charting.Legend legend1 = new System.Windows.Forms.DataVisualization.Charting.Legend();
			System.Windows.Forms.DataVisualization.Charting.Series series1 = new System.Windows.Forms.DataVisualization.Charting.Series();
			this.chart = new System.Windows.Forms.DataVisualization.Charting.Chart();
			this.menuStrip1 = new System.Windows.Forms.MenuStrip();
			this.readBinFileToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.fileOpenMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.openOldVersionFileToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.exportZTraces = new System.Windows.Forms.ToolStripMenuItem();
			this.allToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.selectedToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.exportBeadSelectionToTxtToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.filterBeadsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.beadSelectionToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.label1 = new System.Windows.Forms.Label();
			this.label2 = new System.Windows.Forms.Label();
			this.trackBar = new System.Windows.Forms.TrackBar();
			this.txtRefBead = new System.Windows.Forms.TextBox();
			this.labelNumFrames = new System.Windows.Forms.Label();
			this.label3 = new System.Windows.Forms.Label();
			this.textNumFramesInView = new System.Windows.Forms.TextBox();
			this.label4 = new System.Windows.Forms.Label();
			this.label5 = new System.Windows.Forms.Label();
			this.labelFrame = new System.Windows.Forms.Label();
			this.labelBead = new System.Windows.Forms.Label();
			this.checkUseRef = new System.Windows.Forms.CheckBox();
			this.lutView = new System.Windows.Forms.PictureBox();
			this.splitContainer = new System.Windows.Forms.SplitContainer();
			this.beadListView = new System.Windows.Forms.ListView();
			this.columnHeader_Index = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
			this.columnHeader_Noise = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
			this.columnHeader_Min = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
			this.columnHeader_Max = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
			this.checkX = new System.Windows.Forms.CheckBox();
			this.checkY = new System.Windows.Forms.CheckBox();
			this.checkZ = new System.Windows.Forms.CheckBox();
			this.checkMagnetZ = new System.Windows.Forms.CheckBox();
			this.checkMagnetRot = new System.Windows.Forms.CheckBox();
			((System.ComponentModel.ISupportInitialize)(this.chart)).BeginInit();
			this.menuStrip1.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.trackBar)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.lutView)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.splitContainer)).BeginInit();
			this.splitContainer.Panel1.SuspendLayout();
			this.splitContainer.Panel2.SuspendLayout();
			this.splitContainer.SuspendLayout();
			this.SuspendLayout();
			// 
			// chart
			// 
			this.chart.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom)
						| System.Windows.Forms.AnchorStyles.Left)
						| System.Windows.Forms.AnchorStyles.Right)));
			this.chart.AntiAliasing = System.Windows.Forms.DataVisualization.Charting.AntiAliasingStyles.Text;
			chartArea1.Name = "ChartArea1";
			this.chart.ChartAreas.Add(chartArea1);
			this.chart.IsSoftShadows = false;
			legend1.Name = "Legend1";
			this.chart.Legends.Add(legend1);
			this.chart.Location = new System.Drawing.Point(3, 3);
			this.chart.Name = "chart";
			series1.ChartArea = "ChartArea1";
			series1.Legend = "Legend1";
			series1.Name = "Series1";
			this.chart.Series.Add(series1);
			this.chart.Size = new System.Drawing.Size(687, 352);
			this.chart.SuppressExceptions = true;
			this.chart.TabIndex = 0;
			this.chart.Text = "chart";
			this.chart.TextAntiAliasingQuality = System.Windows.Forms.DataVisualization.Charting.TextAntiAliasingQuality.Normal;
			// 
			// menuStrip1
			// 
			this.menuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.readBinFileToolStripMenuItem,
            this.filterBeadsToolStripMenuItem});
			this.menuStrip1.Location = new System.Drawing.Point(0, 0);
			this.menuStrip1.Name = "menuStrip1";
			this.menuStrip1.Size = new System.Drawing.Size(958, 24);
			this.menuStrip1.TabIndex = 1;
			this.menuStrip1.Text = "menuStrip";
			// 
			// readBinFileToolStripMenuItem
			// 
			this.readBinFileToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.fileOpenMenuItem,
            this.openOldVersionFileToolStripMenuItem,
            this.exportZTraces,
            this.exportBeadSelectionToTxtToolStripMenuItem});
			this.readBinFileToolStripMenuItem.Name = "readBinFileToolStripMenuItem";
			this.readBinFileToolStripMenuItem.Size = new System.Drawing.Size(37, 20);
			this.readBinFileToolStripMenuItem.Text = "File";
			this.readBinFileToolStripMenuItem.Click += new System.EventHandler(this.readBinFileToolStripMenuItem_Click);
			// 
			// fileOpenMenuItem
			// 
			this.fileOpenMenuItem.Name = "fileOpenMenuItem";
			this.fileOpenMenuItem.Size = new System.Drawing.Size(216, 22);
			this.fileOpenMenuItem.Text = "Open";
			this.fileOpenMenuItem.Click += new System.EventHandler(this.menuOpenFile);
			// 
			// openOldVersionFileToolStripMenuItem
			// 
			this.openOldVersionFileToolStripMenuItem.Name = "openOldVersionFileToolStripMenuItem";
			this.openOldVersionFileToolStripMenuItem.Size = new System.Drawing.Size(216, 22);
			this.openOldVersionFileToolStripMenuItem.Text = "Open old version file";
			this.openOldVersionFileToolStripMenuItem.Click += new System.EventHandler(this.openOldVersionFileToolStripMenuItem_Click);
			// 
			// exportZTraces
			// 
			this.exportZTraces.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.allToolStripMenuItem,
            this.selectedToolStripMenuItem});
			this.exportZTraces.Name = "exportZTraces";
			this.exportZTraces.Size = new System.Drawing.Size(216, 22);
			this.exportZTraces.Text = "Export traces to txt";
			// 
			// allToolStripMenuItem
			// 
			this.allToolStripMenuItem.Name = "allToolStripMenuItem";
			this.allToolStripMenuItem.Size = new System.Drawing.Size(152, 22);
			this.allToolStripMenuItem.Text = "All";
			this.allToolStripMenuItem.Click += new System.EventHandler(this.exportTxtAllToolStripMenuItem_Click);
			// 
			// selectedToolStripMenuItem
			// 
			this.selectedToolStripMenuItem.Name = "selectedToolStripMenuItem";
			this.selectedToolStripMenuItem.Size = new System.Drawing.Size(152, 22);
			this.selectedToolStripMenuItem.Text = "Selected";
			this.selectedToolStripMenuItem.Click += new System.EventHandler(this.exportTxtSelectedToolStripMenuItem_Click);
			// 
			// exportBeadSelectionToTxtToolStripMenuItem
			// 
			this.exportBeadSelectionToTxtToolStripMenuItem.Name = "exportBeadSelectionToTxtToolStripMenuItem";
			this.exportBeadSelectionToTxtToolStripMenuItem.Size = new System.Drawing.Size(216, 22);
			this.exportBeadSelectionToTxtToolStripMenuItem.Text = "Export bead selection to txt";
			this.exportBeadSelectionToTxtToolStripMenuItem.Click += new System.EventHandler(this.exportBeadSelectionToTxtToolStripMenuItem_Click);
			// 
			// filterBeadsToolStripMenuItem
			// 
			this.filterBeadsToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.beadSelectionToolStripMenuItem});
			this.filterBeadsToolStripMenuItem.Name = "filterBeadsToolStripMenuItem";
			this.filterBeadsToolStripMenuItem.Size = new System.Drawing.Size(45, 20);
			this.filterBeadsToolStripMenuItem.Text = "Filter";
			// 
			// beadSelectionToolStripMenuItem
			// 
			this.beadSelectionToolStripMenuItem.Name = "beadSelectionToolStripMenuItem";
			this.beadSelectionToolStripMenuItem.Size = new System.Drawing.Size(150, 22);
			this.beadSelectionToolStripMenuItem.Text = "Bead selection";
			this.beadSelectionToolStripMenuItem.Click += new System.EventHandler(this.beadSelectionToolStripMenuItem_Click);
			// 
			// label1
			// 
			this.label1.AutoSize = true;
			this.label1.Location = new System.Drawing.Point(12, 24);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(44, 13);
			this.label1.TabIndex = 2;
			this.label1.Text = "Frames:";
			// 
			// label2
			// 
			this.label2.AutoSize = true;
			this.label2.Location = new System.Drawing.Point(12, 47);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(87, 13);
			this.label2.TabIndex = 3;
			this.label2.Text = "Reference bead:";
			// 
			// trackBar
			// 
			this.trackBar.Location = new System.Drawing.Point(62, 91);
			this.trackBar.Name = "trackBar";
			this.trackBar.Size = new System.Drawing.Size(605, 45);
			this.trackBar.TabIndex = 4;
			this.trackBar.Scroll += new System.EventHandler(this.trackBar_Scroll);
			this.trackBar.ValueChanged += new System.EventHandler(this.trackBar_ValueChanged);
			// 
			// txtRefBead
			// 
			this.txtRefBead.Location = new System.Drawing.Point(105, 44);
			this.txtRefBead.Name = "txtRefBead";
			this.txtRefBead.Size = new System.Drawing.Size(80, 20);
			this.txtRefBead.TabIndex = 5;
			this.txtRefBead.Text = "0";
			this.txtRefBead.KeyDown += new System.Windows.Forms.KeyEventHandler(this.txtRefBead_KeyDown);
			this.txtRefBead.Leave += new System.EventHandler(this.txtRefBead_Leave);
			this.txtRefBead.Validating += new System.ComponentModel.CancelEventHandler(this.txtRefBead_Validating);
			// 
			// labelNumFrames
			// 
			this.labelNumFrames.AutoSize = true;
			this.labelNumFrames.Location = new System.Drawing.Point(102, 24);
			this.labelNumFrames.Name = "labelNumFrames";
			this.labelNumFrames.Size = new System.Drawing.Size(0, 13);
			this.labelNumFrames.TabIndex = 2;
			// 
			// label3
			// 
			this.label3.AutoSize = true;
			this.label3.Location = new System.Drawing.Point(12, 68);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(44, 13);
			this.label3.TabIndex = 6;
			this.label3.Text = "Frames:";
			// 
			// textNumFramesInView
			// 
			this.textNumFramesInView.Location = new System.Drawing.Point(105, 65);
			this.textNumFramesInView.Name = "textNumFramesInView";
			this.textNumFramesInView.Size = new System.Drawing.Size(80, 20);
			this.textNumFramesInView.TabIndex = 7;
			this.textNumFramesInView.Text = "10000";
			this.textNumFramesInView.TextChanged += new System.EventHandler(this.textNumFramesInView_TextChanged);
			// 
			// label4
			// 
			this.label4.AutoSize = true;
			this.label4.Location = new System.Drawing.Point(273, 51);
			this.label4.Name = "label4";
			this.label4.Size = new System.Drawing.Size(35, 13);
			this.label4.TabIndex = 9;
			this.label4.Text = "Bead:";
			// 
			// label5
			// 
			this.label5.AutoSize = true;
			this.label5.Location = new System.Drawing.Point(12, 100);
			this.label5.Name = "label5";
			this.label5.Size = new System.Drawing.Size(39, 13);
			this.label5.TabIndex = 9;
			this.label5.Text = "Frame:";
			// 
			// labelFrame
			// 
			this.labelFrame.AutoSize = true;
			this.labelFrame.Location = new System.Drawing.Point(651, 47);
			this.labelFrame.Name = "labelFrame";
			this.labelFrame.Size = new System.Drawing.Size(0, 13);
			this.labelFrame.TabIndex = 9;
			// 
			// labelBead
			// 
			this.labelBead.AutoSize = true;
			this.labelBead.Location = new System.Drawing.Point(606, 80);
			this.labelBead.Name = "labelBead";
			this.labelBead.Size = new System.Drawing.Size(0, 13);
			this.labelBead.TabIndex = 9;
			// 
			// checkUseRef
			// 
			this.checkUseRef.AutoSize = true;
			this.checkUseRef.Location = new System.Drawing.Point(201, 47);
			this.checkUseRef.Name = "checkUseRef";
			this.checkUseRef.Size = new System.Drawing.Size(66, 17);
			this.checkUseRef.TabIndex = 10;
			this.checkUseRef.Text = "Use ref?";
			this.checkUseRef.UseVisualStyleBackColor = true;
			this.checkUseRef.CheckedChanged += new System.EventHandler(this.checkUseRef_CheckedChanged);
			// 
			// lutView
			// 
			this.lutView.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left)
						| System.Windows.Forms.AnchorStyles.Right)));
			this.lutView.Location = new System.Drawing.Point(773, 30);
			this.lutView.Name = "lutView";
			this.lutView.Size = new System.Drawing.Size(173, 121);
			this.lutView.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
			this.lutView.TabIndex = 11;
			this.lutView.TabStop = false;
			// 
			// splitContainer
			// 
			this.splitContainer.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom)
						| System.Windows.Forms.AnchorStyles.Left)
						| System.Windows.Forms.AnchorStyles.Right)));
			this.splitContainer.Location = new System.Drawing.Point(0, 157);
			this.splitContainer.Name = "splitContainer";
			// 
			// splitContainer.Panel1
			// 
			this.splitContainer.Panel1.Controls.Add(this.beadListView);
			// 
			// splitContainer.Panel2
			// 
			this.splitContainer.Panel2.Controls.Add(this.chart);
			this.splitContainer.Size = new System.Drawing.Size(958, 358);
			this.splitContainer.SplitterDistance = 261;
			this.splitContainer.TabIndex = 12;
			// 
			// beadListView
			// 
			this.beadListView.CheckBoxes = true;
			this.beadListView.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader_Index,
            this.columnHeader_Noise,
            this.columnHeader_Min,
            this.columnHeader_Max});
			this.beadListView.Dock = System.Windows.Forms.DockStyle.Fill;
			this.beadListView.FullRowSelect = true;
			this.beadListView.Location = new System.Drawing.Point(0, 0);
			this.beadListView.Name = "beadListView";
			this.beadListView.Size = new System.Drawing.Size(261, 358);
			this.beadListView.TabIndex = 0;
			this.beadListView.UseCompatibleStateImageBehavior = false;
			this.beadListView.View = System.Windows.Forms.View.Details;
			this.beadListView.SelectedIndexChanged += new System.EventHandler(this.beadListView_SelectedIndexChanged);
			// 
			// columnHeader_Index
			// 
			this.columnHeader_Index.Text = "Index";
			// 
			// columnHeader_Noise
			// 
			this.columnHeader_Noise.Text = "Noise";
			this.columnHeader_Noise.Width = 109;
			// 
			// columnHeader_Min
			// 
			this.columnHeader_Min.Text = "Min";
			// 
			// columnHeader_Max
			// 
			this.columnHeader_Max.Text = "Max";
			// 
			// checkX
			// 
			this.checkX.AutoSize = true;
			this.checkX.Location = new System.Drawing.Point(695, 31);
			this.checkX.Name = "checkX";
			this.checkX.Size = new System.Drawing.Size(33, 17);
			this.checkX.TabIndex = 13;
			this.checkX.Text = "X";
			this.checkX.UseVisualStyleBackColor = true;
			this.checkX.CheckedChanged += new System.EventHandler(this.checkY_CheckedChanged);
			// 
			// checkY
			// 
			this.checkY.AutoSize = true;
			this.checkY.Location = new System.Drawing.Point(694, 54);
			this.checkY.Name = "checkY";
			this.checkY.Size = new System.Drawing.Size(33, 17);
			this.checkY.TabIndex = 13;
			this.checkY.Text = "Y";
			this.checkY.UseVisualStyleBackColor = true;
			this.checkY.CheckedChanged += new System.EventHandler(this.checkY_CheckedChanged);
			// 
			// checkZ
			// 
			this.checkZ.AutoSize = true;
			this.checkZ.Checked = true;
			this.checkZ.CheckState = System.Windows.Forms.CheckState.Checked;
			this.checkZ.Location = new System.Drawing.Point(694, 77);
			this.checkZ.Name = "checkZ";
			this.checkZ.Size = new System.Drawing.Size(33, 17);
			this.checkZ.TabIndex = 13;
			this.checkZ.Text = "Z";
			this.checkZ.UseVisualStyleBackColor = true;
			this.checkZ.CheckedChanged += new System.EventHandler(this.checkY_CheckedChanged);
			// 
			// checkMagnetZ
			// 
			this.checkMagnetZ.AutoSize = true;
			this.checkMagnetZ.Location = new System.Drawing.Point(695, 100);
			this.checkMagnetZ.Name = "checkMagnetZ";
			this.checkMagnetZ.Size = new System.Drawing.Size(72, 17);
			this.checkMagnetZ.TabIndex = 13;
			this.checkMagnetZ.Text = "Magnet Z";
			this.checkMagnetZ.UseVisualStyleBackColor = true;
			this.checkMagnetZ.CheckedChanged += new System.EventHandler(this.checkMagnetZ_CheckedChanged);
			// 
			// checkMagnetRot
			// 
			this.checkMagnetRot.AutoSize = true;
			this.checkMagnetRot.Location = new System.Drawing.Point(694, 123);
			this.checkMagnetRot.Name = "checkMagnetRot";
			this.checkMagnetRot.Size = new System.Drawing.Size(73, 17);
			this.checkMagnetRot.TabIndex = 13;
			this.checkMagnetRot.Text = "Magnet R";
			this.checkMagnetRot.UseVisualStyleBackColor = true;
			this.checkMagnetRot.CheckedChanged += new System.EventHandler(this.checkMagnetZ_CheckedChanged);
			// 
			// ViewTraceDlg
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(958, 516);
			this.Controls.Add(this.checkZ);
			this.Controls.Add(this.checkY);
			this.Controls.Add(this.lutView);
			this.Controls.Add(this.checkMagnetRot);
			this.Controls.Add(this.checkMagnetZ);
			this.Controls.Add(this.checkX);
			this.Controls.Add(this.splitContainer);
			this.Controls.Add(this.checkUseRef);
			this.Controls.Add(this.labelFrame);
			this.Controls.Add(this.label5);
			this.Controls.Add(this.labelBead);
			this.Controls.Add(this.label4);
			this.Controls.Add(this.textNumFramesInView);
			this.Controls.Add(this.label3);
			this.Controls.Add(this.txtRefBead);
			this.Controls.Add(this.trackBar);
			this.Controls.Add(this.label2);
			this.Controls.Add(this.labelNumFrames);
			this.Controls.Add(this.label1);
			this.Controls.Add(this.menuStrip1);
			this.MainMenuStrip = this.menuStrip1;
			this.Name = "ViewTraceDlg";
			this.Text = "Bead Tracker Trace Viewer";
			((System.ComponentModel.ISupportInitialize)(this.chart)).EndInit();
			this.menuStrip1.ResumeLayout(false);
			this.menuStrip1.PerformLayout();
			((System.ComponentModel.ISupportInitialize)(this.trackBar)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.lutView)).EndInit();
			this.splitContainer.Panel1.ResumeLayout(false);
			this.splitContainer.Panel2.ResumeLayout(false);
			((System.ComponentModel.ISupportInitialize)(this.splitContainer)).EndInit();
			this.splitContainer.ResumeLayout(false);
			this.ResumeLayout(false);
			this.PerformLayout();

		}

		#endregion

		private System.Windows.Forms.DataVisualization.Charting.Chart chart;
		private System.Windows.Forms.MenuStrip menuStrip1;
		private System.Windows.Forms.ToolStripMenuItem readBinFileToolStripMenuItem;
		private System.Windows.Forms.ToolStripMenuItem fileOpenMenuItem;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.TrackBar trackBar;
		private System.Windows.Forms.TextBox txtRefBead;
		private System.Windows.Forms.Label labelNumFrames;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.TextBox textNumFramesInView;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.Label label5;
		private System.Windows.Forms.Label labelFrame;
		private System.Windows.Forms.Label labelBead;
		private System.Windows.Forms.ToolStripMenuItem openOldVersionFileToolStripMenuItem;
		private System.Windows.Forms.ToolStripMenuItem exportZTraces;
		private System.Windows.Forms.CheckBox checkUseRef;
		private System.Windows.Forms.PictureBox lutView;
		private System.Windows.Forms.SplitContainer splitContainer;
		private System.Windows.Forms.CheckBox checkX;
		private System.Windows.Forms.CheckBox checkY;
		private System.Windows.Forms.CheckBox checkZ;
		private System.Windows.Forms.CheckBox checkMagnetZ;
		private System.Windows.Forms.ListView beadListView;
		private System.Windows.Forms.ColumnHeader columnHeader_Index;
		private System.Windows.Forms.ColumnHeader columnHeader_Noise;
		private System.Windows.Forms.ToolStripMenuItem filterBeadsToolStripMenuItem;
		private System.Windows.Forms.ToolStripMenuItem beadSelectionToolStripMenuItem;
		private System.Windows.Forms.ColumnHeader columnHeader_Min;
		private System.Windows.Forms.ColumnHeader columnHeader_Max;
		private System.Windows.Forms.ToolStripMenuItem exportBeadSelectionToTxtToolStripMenuItem;
		private System.Windows.Forms.ToolStripMenuItem allToolStripMenuItem;
		private System.Windows.Forms.ToolStripMenuItem selectedToolStripMenuItem;
		private System.Windows.Forms.CheckBox checkMagnetRot;

	}
}

