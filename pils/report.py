"""
PDF Report Generator for PILS Flight Data

This module provides tools to generate comprehensive PDF reports from flight data,
including plots, statistics, and analysis summaries.
"""

import os
from pathlib import Path
from io import BytesIO
import base64
from datetime import datetime
from typing import Optional, List, Dict, Any, Union

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

try:
    from jinja2 import Template
    from weasyprint import HTML
    import markdown
    REPORT_DEPENDENCIES_AVAILABLE = True
except ImportError:
    REPORT_DEPENDENCIES_AVAILABLE = False
    print("[Report] Warning: jinja2, weasyprint, or markdown not installed. Install with: pip install jinja2 weasyprint markdown")


class FlightReport:
    """
    Generate PDF reports for flight data analysis.
    
    Examples
    --------
    >>> from pils import DataHandler, FlightReport
    >>> d = DataHandler('/path/to/flight')
    >>> d.load_data()
    >>> 
    >>> report = FlightReport(d)
    >>> report.add_section("GPS Data", "Analysis of GPS coordinates")
    >>> report.add_plot_from_data('gps', x='datetime', y='latitude', title='GPS Latitude')
    >>> report.generate('flight_report.pdf')
    """
    
    def __init__(self, datahandler, title: Optional[str] = None):
        """
        Initialize FlightReport with a DataHandler instance.
        
        Parameters
        ----------
        datahandler : DataHandler
            DataHandler instance with loaded flight data
        title : str, optional
            Report title. If None, auto-generated from flight name.
        """
        if not REPORT_DEPENDENCIES_AVAILABLE:
            raise ImportError(
                "Report generation requires additional dependencies. "
                "Install with: pip install jinja2 weasyprint markdown"
            )
        
        self.datahandler = datahandler
        self.title = title or f"Flight Data Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        self.sections = []
        
    def add_section(self, heading: str, content: str):
        """
        Add a text section to the report.
        
        Parameters
        ----------
        heading : str
            Section heading (e.g., "Summary", "Analysis")
        content : str
            Section content (supports Markdown)
        """
        self.sections.append({
            'type': 'text',
            'heading': heading,
            'content': content
        })
    
    def add_plot(self, fig, caption: str = ""):
        """
        Add a matplotlib figure to the report.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Matplotlib figure to add
        caption : str, optional
            Caption for the plot
        """
        # Convert figure to base64
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        
        self.sections.append({
            'type': 'plot',
            'image': image_base64,
            'caption': caption
        })
    
    def add_plot_from_data(self, sensor: str, x: str, y: Union[str, List[str]], 
                          title: str = "", xlabel: str = "", ylabel: str = "",
                          caption: str = "", **plot_kwargs):
        """
        Create and add a plot from sensor data.
        
        Parameters
        ----------
        sensor : str
            Sensor name (e.g., 'gps', 'adc', 'drone')
        x : str
            Column name for x-axis
        y : str or list of str
            Column name(s) for y-axis
        title : str, optional
            Plot title
        xlabel : str, optional
            X-axis label
        ylabel : str, optional
            Y-axis label
        caption : str, optional
            Caption for the plot
        **plot_kwargs
            Additional arguments passed to plt.plot()
        
        Examples
        --------
        >>> report.add_plot_from_data('gps', 'datetime', 'latitude', 
        ...                           title='GPS Latitude', ylabel='Latitude (deg)')
        >>> report.add_plot_from_data('inclino', 'datetime', ['pitch', 'roll', 'yaw'],
        ...                           title='Inclinometer Angles')
        """
        if sensor not in self.datahandler:
            print(f"[Report] Warning: Sensor '{sensor}' not found in data")
            return
        
        data = self.datahandler[sensor]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if isinstance(y, (list, tuple)):
            for y_col in y:
                ax.plot(data[x], data[y_col], label=y_col, **plot_kwargs)
            ax.legend()
        else:
            ax.plot(data[x], data[y], **plot_kwargs)
        
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        elif x:
            ax.set_xlabel(x)
        if ylabel:
            ax.set_ylabel(ylabel)
        elif isinstance(y, str):
            ax.set_ylabel(y)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        self.add_plot(fig, caption)
    
    def add_statistics_table(self, sensor: str, columns: Optional[List[str]] = None):
        """
        Add a statistics table for sensor data.
        
        Parameters
        ----------
        sensor : str
            Sensor name
        columns : list of str, optional
            Columns to include statistics for. If None, uses all numeric columns.
        """
        if sensor not in self.datahandler:
            print(f"[Report] Warning: Sensor '{sensor}' not found in data")
            return
        
        data = self.datahandler[sensor]
        
        if columns is None:
            # Get numeric columns
            columns = data.select_dtypes(include=['number']).columns.tolist()
        
        stats = data[columns].describe()
        
        # Convert to markdown table
        table_md = f"\n### {sensor.upper()} Statistics\n\n"
        table_md += stats.to_markdown()
        
        self.sections.append({
            'type': 'text',
            'heading': f'{sensor.upper()} Statistics',
            'content': table_md
        })
    
    def add_data_summary(self):
        """
        Add an automatic summary of all available sensors and data counts.
        """
        summary = "## Available Sensors\n\n"
        
        for sensor_name in self.datahandler.keys():
            data = self.datahandler[sensor_name]
            if hasattr(data, '__len__'):
                summary += f"- **{sensor_name}**: {len(data)} records\n"
            else:
                summary += f"- **{sensor_name}**: Available\n"
        
        self.sections.append({
            'type': 'text',
            'heading': 'Data Summary',
            'content': summary
        })
    
    def generate(self, output_path: Optional[str] = None) -> str:
        """
        Generate the PDF report.
        
        Parameters
        ----------
        output_path : str, optional
            Output PDF path. If None, saves to ~/flight_report_YYYYMMDD_HHMMSS.pdf
        
        Returns
        -------
        str
            Path to the generated PDF file
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.expanduser(f"~/flight_report_{timestamp}.pdf")
        
        # Build markdown content
        md_content = f"# {self.title}\n\n"
        md_content += f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        md_content += "---\n\n"
        
        for section in self.sections:
            if section['type'] == 'text':
                md_content += f"## {section['heading']}\n\n"
                md_content += f"{section['content']}\n\n"
            elif section['type'] == 'plot':
                if section['caption']:
                    md_content += f"## {section['caption']}\n\n"
                md_content += f"![Chart](data:image/png;base64,{section['image']})\n\n"
        
        # Convert Markdown -> HTML
        html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
        
        # Add CSS styling
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                @page {{
                    margin: 2cm;
                    size: A4;
                }}
                body {{
                    font-family: 'Helvetica', 'Arial', sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 100%;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    border-bottom: 1px solid #bdc3c7;
                    padding-bottom: 5px;
                    margin-top: 30px;
                }}
                h3 {{
                    color: #7f8c8d;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 20px auto;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                code {{
                    background-color: #f4f4f4;
                    padding: 2px 5px;
                    border-radius: 3px;
                }}
                hr {{
                    border: none;
                    border-top: 2px solid #3498db;
                    margin: 30px 0;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Generate PDF
        HTML(string=styled_html).write_pdf(output_path)
        
        print(f"[Report] PDF generated: {output_path}")
        return output_path


def quick_report(datahandler, output_path: Optional[str] = None, 
                sensors: Optional[List[str]] = None) -> str:
    """
    Generate a quick report with standard plots for all sensors.
    
    Parameters
    ----------
    datahandler : DataHandler
        DataHandler instance with loaded flight data
    output_path : str, optional
        Output PDF path
    sensors : list of str, optional
        List of sensors to include. If None, includes all available sensors.
    
    Returns
    -------
    str
        Path to the generated PDF file
    
    Examples
    --------
    >>> from pils import DataHandler, quick_report
    >>> d = DataHandler('/path/to/flight')
    >>> d.load_data()
    >>> quick_report(d, 'my_report.pdf')
    """
    report = FlightReport(datahandler)
    
    # Add data summary
    report.add_data_summary()
    
    # Determine which sensors to plot
    if sensors is None:
        sensors = list(datahandler.keys())
    
    # Add plots for each sensor
    plot_configs = {
        'gps': [
            {'y': 'latitude', 'title': 'GPS Latitude', 'ylabel': 'Latitude (deg)'},
            {'y': 'longitude', 'title': 'GPS Longitude', 'ylabel': 'Longitude (deg)'},
            {'y': 'altitude', 'title': 'GPS Altitude', 'ylabel': 'Altitude (m)'},
        ],
        'inclino': [
            {'y': ['pitch', 'roll', 'yaw'], 'title': 'Inclinometer Angles', 
             'ylabel': 'Angle (deg)'},
        ],
        'adc': [
            {'y': list(range(4)), 'title': 'ADC Channels', 'ylabel': 'Voltage (V)'},
        ],
        'drone': [
            {'y': 'altitude', 'title': 'Drone Altitude', 'ylabel': 'Altitude (m)'},
        ],
        'baro': [
            {'y': 'pressure', 'title': 'Barometric Pressure', 'ylabel': 'Pressure (hPa)'},
        ],
    }
    
    for sensor in sensors:
        if sensor in datahandler and sensor in plot_configs:
            report.add_section(f"{sensor.upper()} Data", 
                             f"Analysis of {sensor} sensor data")
            
            data = datahandler[sensor]
            x_col = 'datetime' if 'datetime' in data.columns else data.columns[0]
            
            for config in plot_configs[sensor]:
                try:
                    report.add_plot_from_data(sensor, x=x_col, **config)
                except Exception as e:
                    print(f"[Report] Warning: Could not create plot for {sensor}: {e}")
    
    return report.generate(output_path)
