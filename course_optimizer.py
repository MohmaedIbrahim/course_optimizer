import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pulp
import numpy as np
from typing import Dict, List, Tuple
import io
from scipy.cluster.hierarchy import dendrogram, linkage

class CourseCoveringProblem:
    """Course covering optimization problem solver matching exact mathematical formulation."""
    
    def __init__(self, courses: List[str], professors: List[str], terms: List[str],
                 course_preferences: Dict[Tuple[str, str], float],
                 term_preferences: Dict[Tuple[str, str], float],
                 course_streams: Dict[Tuple[str, str], int],
                 professor_total_load: Dict[str, int],
                 professor_term_limits: Dict[Tuple[str, str], int],
                 course_offerings: Dict[Tuple[str, str], int]):
        
        self.courses = courses
        self.professors = professors
        self.terms = terms
        self.course_preferences = course_preferences
        self.term_preferences = term_preferences
        self.course_streams = course_streams
        self.professor_total_load = professor_total_load
        self.professor_term_limits = professor_term_limits
        self.course_offerings = course_offerings
        
        self.model = None
        self.x_vars = {}
        
        self._validate_preferences()
        
    def _validate_preferences(self):
        """Validate that preference scores are in 0-10 range."""
        for (course, prof), score in self.course_preferences.items():
            if not (0 <= score <= 10):
                raise ValueError(f"Course preference c_{{{course},{prof}}} = {score} must be in range [0,10]")
        
        for (prof, term), score in self.term_preferences.items():
            if not (0 <= score <= 10):
                raise ValueError(f"Term preference t_{{{prof},{term}}} = {score} must be in range [0,10]")
    
    def build_model(self):
        """Build the optimization model matching the mathematical formulation exactly."""
        self.model = pulp.LpProblem("Course_Covering_Mathematical", pulp.LpMaximize)
        
        self.x_vars = {}
        for course in self.courses:
            for professor in self.professors:
                for term in self.terms:
                    if self.course_offerings.get((course, term), 0) == 1:
                        self.x_vars[(course, professor, term)] = pulp.LpVariable(
                            f"x_{course}_{professor}_{term}", cat='Binary'
                        )
        
        course_pref_term = pulp.lpSum([
            self.course_preferences.get((course, professor), 0) * self.x_vars[(course, professor, term)]
            for course in self.courses
            for professor in self.professors
            for term in self.terms
            if (course, professor, term) in self.x_vars
        ])
        
        term_pref_term = pulp.lpSum([
            self.term_preferences.get((professor, term), 0) * self.x_vars[(course, professor, term)]
            for course in self.courses
            for professor in self.professors
            for term in self.terms
            if (course, professor, term) in self.x_vars
        ])
        
        self.model += course_pref_term + term_pref_term
        self._add_constraints()
        
    def _add_constraints(self):
        """Add all constraints matching the mathematical formulation exactly."""
        
        for professor in self.professors:
            for term in self.terms:
                term_stream_load = pulp.lpSum([
                    self.course_streams.get((course, term), 0) * self.x_vars[(course, professor, term)]
                    for course in self.courses
                    if (course, professor, term) in self.x_vars
                ])
                max_streams_in_term = self.professor_term_limits.get((professor, term), 0)
                self.model += (
                    term_stream_load <= max_streams_in_term,
                    f"StreamLoad_L_{professor}_{term}"
                )
        
        for professor in self.professors:
            total_courses_assigned = pulp.lpSum([
                self.x_vars[(course, professor, term)]
                for course in self.courses
                for term in self.terms
                if (course, professor, term) in self.x_vars
            ])
            self.model += (
                total_courses_assigned <= self.professor_total_load[professor],
                f"TotalLoad_b_{professor}"
            )
        
        for course in self.courses:
            for term in self.terms:
                if self.course_offerings.get((course, term), 0) == 1:
                    self.model += (
                        pulp.lpSum([
                            self.x_vars[(course, professor, term)]
                            for professor in self.professors
                            if (course, professor, term) in self.x_vars
                        ]) == 1,
                        f"CourseOffering_O_{course}_{term}"
                    )
    
    def solve(self):
        """Solve the optimization problem."""
        if self.model is None:
            self.build_model()
        
        solver = pulp.PULP_CBC_CMD(msg=0)
        self.model.solve(solver)
        
        status = pulp.LpStatus[self.model.status]
        
        if status == 'Optimal':
            return self._extract_solution()
        else:
            return {
                'status': status,
                'objective_value': None,
                'assignments': {},
                'professor_loads': {},
                'unassigned_offerings': [],
                'constraint_violations': self._analyze_infeasibility() if status == 'Infeasible' else None
            }
    
    def _extract_solution(self):
        """Extract solution from solved model."""
        assignments = {}
        professor_loads = {
            prof: {
                'total_courses': 0,
                'streams_per_term': {term: 0 for term in self.terms},
                'total_streams': 0
            } 
            for prof in self.professors
        }
        unassigned_offerings = []
        
        for course in self.courses:
            for term in self.terms:
                if self.course_offerings.get((course, term), 0) == 1:
                    assigned = False
                    for professor in self.professors:
                        if (course, professor, term) in self.x_vars and self.x_vars[(course, professor, term)].varValue == 1:
                            assignments[(course, term)] = professor
                            professor_loads[professor]['total_courses'] += 1
                            streams_count = self.course_streams.get((course, term), 1)
                            professor_loads[professor]['streams_per_term'][term] += streams_count
                            professor_loads[professor]['total_streams'] += streams_count
                            assigned = True
                            break
                    
                    if not assigned:
                        unassigned_offerings.append((course, term))
        
        return {
            'status': 'Optimal',
            'objective_value': pulp.value(self.model.objective),
            'assignments': assignments,
            'professor_loads': professor_loads,
            'unassigned_offerings': unassigned_offerings
        }
    
    def _analyze_infeasibility(self):
        """Analyze potential constraint violations if problem is infeasible."""
        violations = []
        
        total_required_streams = sum([
            self.course_streams.get((course, term), 1)
            for course in self.courses
            for term in self.terms
            if self.course_offerings.get((course, term), 0) == 1
        ])
        
        total_available_streams = sum([
            self.professor_term_limits.get((prof, term), 0)
            for prof in self.professors
            for term in self.terms
        ])
        
        if total_required_streams > total_available_streams:
            violations.append(f"Total required streams ({total_required_streams}) exceed total available capacity ({total_available_streams})")
        
        total_required_courses = sum([
            1 for course in self.courses
            for term in self.terms
            if self.course_offerings.get((course, term), 0) == 1
        ])
        
        total_available_courses = sum(self.professor_total_load.values())
        
        if total_required_courses > total_available_courses:
            violations.append(f"Total required courses ({total_required_courses}) exceed total available slots ({total_available_courses})")
        
        return violations


def main():
    st.set_page_config(
        page_title="RASTA-OP",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("RASTA-OP")
    st.markdown("Optimize faculty assignments using exact mathematical formulation with L_jk and b_j constraints")
    st.markdown("---")
    
    input_method = st.sidebar.selectbox(
        "Choose Input Method:",
        ["Manual Input", "Excel Upload"]
    )
    
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'courses' not in st.session_state:
        st.session_state.courses = []
    if 'professors' not in st.session_state:
        st.session_state.professors = []
    if 'terms' not in st.session_state:
        st.session_state.terms = ['T1', 'T2', 'T3']
    
    if input_method == "Excel Upload":
        if st.session_state.step == 1:
            show_excel_upload_step()
        elif st.session_state.step == 2:
            show_data_analysis_step()
        elif st.session_state.step == 3:
            show_clustering_analysis_step()
        elif st.session_state.step == 4:
            show_results_step()
    else:
        if st.session_state.step == 1:
            show_setup_step()
        elif st.session_state.step == 2:
            show_constraints_step()
        elif st.session_state.step == 3:
            show_preferences_step()
        elif st.session_state.step == 4:
            show_clustering_analysis_step()
        elif st.session_state.step == 5:
            show_results_step()


def show_setup_step():
    """Show the setup step for courses and professors."""
    st.header("Step 1: Setup Courses and Professors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Courses")
        courses_input = st.text_area(
            "Enter courses (one per line):",
            value="ACTL1\nACTL2\nACTL3\nACTL4\nACTL5",
            height=120
        )
    
    with col2:
        st.subheader("Professors")
        professors_input = st.text_area(
            "Enter professors (one per line):",
            value="Jonathan\nJK\nPatrick\nAndres",
            height=120
        )
    
    if st.button("Next: Set Constraints", type="primary"):
        courses = [course.strip() for course in courses_input.split('\n') if course.strip()]
        professors = [prof.strip() for prof in professors_input.split('\n') if prof.strip()]
        
        if not courses or not professors:
            st.error("Please enter at least one course and one professor.")
        else:
            st.session_state.courses = courses
            st.session_state.professors = professors
            st.session_state.step = 2
            st.rerun()


def show_constraints_step():
    """Show constraints configuration step."""
    st.header("Step 2: Configure Constraints")
    
    courses = st.session_state.courses
    professors = st.session_state.professors
    terms = st.session_state.terms
    
    st.subheader("Course Offerings & Streams (O_ik, n_ik)")
    st.markdown("Configure which courses are offered in which terms and how many streams each has")
    
    course_offerings = {}
    course_streams = {}
    
    for course in courses:
        st.write(f"**{course}**")
        cols = st.columns(len(terms))
        
        for idx, term in enumerate(terms):
            with cols[idx]:
                offered = st.checkbox(f"Offer in {term}", key=f"offer_{course}_{term}")
                course_offerings[(course, term)] = 1 if offered else 0
                
                if offered:
                    streams = st.number_input(
                        f"Streams in {term}:",
                        min_value=1, max_value=5, value=1,
                        key=f"streams_{course}_{term}"
                    )
                    course_streams[(course, term)] = streams
                else:
                    course_streams[(course, term)] = 0
    
    st.subheader("Professor Constraints")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Total Teaching Load (b_j)**")
        st.markdown("Maximum total courses per professor across all terms")
        professor_total_load = {}
        
        for professor in professors:
            load = st.number_input(
                f"{professor} - Total courses (b_j):",
                min_value=1, max_value=10, value=4,
                key=f"total_load_{professor}"
            )
            professor_total_load[professor] = load
    
    with col2:
        st.write("**Term Stream Limits (L_jk)**")
        st.markdown("Maximum streams per professor per term (independent from b_j)")
        professor_term_limits = {}
        
        for professor in professors:
            st.write(f"**{professor}:**")
            sub_cols = st.columns(len(terms))
            
            for idx, term in enumerate(terms):
                with sub_cols[idx]:
                    limit = st.number_input(
                        f"{term}:",
                        min_value=0, max_value=6, value=2,
                        key=f"term_limit_{professor}_{term}",
                        help=f"Max streams {professor} can teach in {term}"
                    )
                    professor_term_limits[(professor, term)] = limit
    
    st.subheader("Constraint Summary")
    
    st.write("**Course Offerings Matrix (O_ik):**")
    offerings_data = []
    for course in courses:
        row = {'Course': course}
        for term in terms:
            if course_offerings.get((course, term), 0) == 1:
                streams = course_streams.get((course, term), 1)
                row[term] = f"✓ ({streams} streams)"
            else:
                row[term] = "✗"
        offerings_data.append(row)
    
    offerings_df = pd.DataFrame(offerings_data)
    st.dataframe(offerings_df, hide_index=True)
    
    st.write("**Professor Constraints Summary:**")
    prof_summary = []
    for prof in professors:
        row = {'Professor': prof, 'Total Load (b_j)': professor_total_load[prof]}
        for term in terms:
            row[f"{term} Limit (L_jk)"] = professor_term_limits[(prof, term)]
        prof_summary.append(row)
    
    prof_df = pd.DataFrame(prof_summary)
    st.dataframe(prof_df, hide_index=True)
    
    st.session_state.course_offerings = course_offerings
    st.session_state.course_streams = course_streams
    st.session_state.professor_total_load = professor_total_load
    st.session_state.professor_term_limits = professor_term_limits
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Setup"):
            st.session_state.step = 1
            st.rerun()
    
    with col2:
        if st.button("Next: Set Preferences", type="primary"):
            st.session_state.step = 3
            st.rerun()


def show_preferences_step():
    """Show preferences configuration step."""
    st.header("Step 3: Set Preferences (0-10 Scale)")
    
    courses = st.session_state.courses
    professors = st.session_state.professors
    terms = st.session_state.terms
    
    st.subheader("Course Preferences (c_ij)")
    st.markdown("**Scale: 0 = Cannot teach, 5 = Neutral, 10 = Strongly prefer**")
    
    course_preferences = {}
    
    for professor in professors:
        st.write(f"**{professor}'s Course Preferences:**")
        cols = st.columns(min(4, len(courses)))
        
        for idx, course in enumerate(courses):
            with cols[idx % len(cols)]:
                pref = st.slider(
                    f"{course}",
                    min_value=0, max_value=10, value=5,
                    key=f"course_pref_{course}_{professor}"
                )
                course_preferences[(course, professor)] = pref
    
    st.subheader("Term Preferences (t_jk)")
    st.markdown("**Scale: 0 = Cannot teach, 5 = Neutral, 10 = Strongly prefer**")
    
    term_preferences = {}
    
    for professor in professors:
        st.write(f"**{professor}'s Term Preferences:**")
        cols = st.columns(len(terms))
        
        for idx, term in enumerate(terms):
            with cols[idx]:
                pref = st.slider(
                    f"{term}",
                    min_value=0, max_value=10, value=5,
                    key=f"term_pref_{professor}_{term}"
                )
                term_preferences[(professor, term)] = pref
    
    st.subheader("Preference Matrices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Course Preferences (c_ij):**")
        course_pref_matrix = pd.DataFrame(index=courses, columns=professors)
        for course in courses:
            for prof in professors:
                course_pref_matrix.loc[course, prof] = course_preferences.get((course, prof), 0)
        st.dataframe(course_pref_matrix)
    
    with col2:
        st.write("**Term Preferences (t_jk):**")
        term_pref_matrix = pd.DataFrame(index=professors, columns=terms)
        for prof in professors:
            for term in terms:
                term_pref_matrix.loc[prof, term] = term_preferences.get((prof, term), 0)
        st.dataframe(term_pref_matrix)
    
    st.session_state.course_preferences = course_preferences
    st.session_state.term_preferences = term_preferences
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Constraints"):
            st.session_state.step = 2
            st.rerun()
    
    with col2:
        if st.button("Next: Clustering Analysis", type="primary"):
            st.session_state.step = 4
            st.rerun()


def show_excel_upload_step():
    """Show Excel upload interface."""
    st.header("Excel File Upload")
    
    st.subheader("Required Excel File Format")
    st.markdown("""
    Your Excel file must contain exactly 4 sheets with this structure:
    
    **Sheet 1: c_ij (Course Preferences)**
    - Staff names in column A starting from A3
    - Course names in row 1 starting from B1
    - Course codes in row 2 starting from B2
    - Preference scores (0-10) in the data area
    
    **Sheet 2: t_jk, L_jk, b_j (Professor Constraints)**
    - Staff names in column A starting from A3
    - t_jk (Term Preferences): 3 columns for T1, T2, T3 (0-10 scale)
    - L_jk (Term Max Load): 3 columns for T1, T2, T3 (max streams per term)
    - b_j (Total Teaching Load): 1 column for total courses
    
    **Sheet 3: O_jk (Course Offerings)**
    - Course codes in column A
    - T1, T2, T3 columns with 1/0 values (1 = offered, 0 = not offered)
    
    **Sheet 4: n_jk (Course Streams)**
    - Course codes in column A  
    - T1, T2, T3 columns with stream counts
    """)
    
    uploaded_file = st.file_uploader(
        "Upload your Excel file:",
        type=['xlsx', 'xls'],
        help="Upload an Excel file with the 4 required sheets"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner("Reading Excel file..."):
                excel_data = pd.ExcelFile(uploaded_file)
                
                required_sheets = ['c_ij', 't_jk_L_jk_b_j', 'O_jk', 'n_jk']
                available_sheets = excel_data.sheet_names
                
                st.write("Available sheets:", available_sheets)
                
                sheet_mapping = {}
                for req_sheet in required_sheets:
                    matched = False
                    for avail_sheet in available_sheets:
                        if req_sheet.lower() in avail_sheet.lower() or avail_sheet.lower() in req_sheet.lower():
                            sheet_mapping[req_sheet] = avail_sheet
                            matched = True
                            break
                    if not matched:
                        if req_sheet == 'c_ij' and len(available_sheets) > 0:
                            sheet_mapping[req_sheet] = available_sheets[0]
                        elif req_sheet == 't_jk_L_jk_b_j' and len(available_sheets) > 1:
                            sheet_mapping[req_sheet] = available_sheets[1]
                        elif req_sheet == 'O_jk' and len(available_sheets) > 2:
                            sheet_mapping[req_sheet] = available_sheets[2]
                        elif req_sheet == 'n_jk' and len(available_sheets) > 3:
                            sheet_mapping[req_sheet] = available_sheets[3]
                
                if len(sheet_mapping) != 4:
                    st.error(f"Could not find all required sheets. Found: {list(sheet_mapping.values())}")
                    return
                
                sheet1 = pd.read_excel(uploaded_file, sheet_name=sheet_mapping['c_ij'], header=None)
                sheet2 = pd.read_excel(uploaded_file, sheet_name=sheet_mapping['t_jk_L_jk_b_j'])
                sheet3 = pd.read_excel(uploaded_file, sheet_name=sheet_mapping['O_jk'])
                sheet4 = pd.read_excel(uploaded_file, sheet_name=sheet_mapping['n_jk'])
                
                st.subheader("Processing Course Preferences (c_ij)")
                
                course_names = []
                course_codes = []
                for col_idx in range(1, len(sheet1.columns)):
                    if pd.notna(sheet1.iloc[0, col_idx]):
                        course_names.append(str(sheet1.iloc[0, col_idx]).strip())
                    if pd.notna(sheet1.iloc[1, col_idx]):
                        course_codes.append(str(sheet1.iloc[1, col_idx]).strip())
                
                staff_names = []
                for row_idx in range(2, len(sheet1)):
                    if pd.notna(sheet1.iloc[row_idx, 0]):
                        staff_names.append(str(sheet1.iloc[row_idx, 0]).strip())
                
                course_preferences = {}
                for staff_idx, staff in enumerate(staff_names):
                    for course_idx, course in enumerate(course_codes):
                        if course_idx < len(course_codes):
                            row_idx = staff_idx + 2
                            col_idx = course_idx + 1
                            if row_idx < len(sheet1) and col_idx < len(sheet1.columns):
                                pref_val = sheet1.iloc[row_idx, col_idx]
                                if pd.notna(pref_val):
                                    course_preferences[(course, staff)] = float(pref_val)
                                else:
                                    course_preferences[(course, staff)] = 0.0
                
                st.success(f"Found {len(staff_names)} staff and {len(course_codes)} courses")
                
                terms = ['T1', 'T2', 'T3']
                term_preferences = {}
                professor_term_limits = {}
                professor_total_load = {}
                
                for idx, row in sheet2.iterrows():
                    if pd.notna(row.iloc[0]):
                        staff = str(row.iloc[0]).strip()
                        
                        for i, term in enumerate(terms):
                            if len(row) > i + 1 and pd.notna(row.iloc[i + 1]):
                                term_preferences[(staff, term)] = float(row.iloc[i + 1])
                            else:
                                term_preferences[(staff, term)] = 5.0
                        
                        for i, term in enumerate(terms):
                            if len(row) > i + 4 and pd.notna(row.iloc[i + 4]):
                                professor_term_limits[(staff, term)] = int(row.iloc[i + 4])
                            else:
                                professor_term_limits[(staff, term)] = 2
                        
                        if len(row) > 7 and pd.notna(row.iloc[7]):
                            professor_total_load[staff] = int(row.iloc[7])
                        else:
                            professor_total_load[staff] = 4
                
                course_offerings = {}
                for idx, row in sheet3.iterrows():
                    if pd.notna(row.iloc[0]):
                        course = str(row.iloc[0]).strip()
                        for i, term in enumerate(terms):
                            if len(row) > i + 1 and pd.notna(row.iloc[i + 1]):
                                course_offerings[(course, term)] = int(row.iloc[i + 1])
                            else:
                                course_offerings[(course, term)] = 0
                
                course_streams = {}
                for idx, row in sheet4.iterrows():
                    if pd.notna(row.iloc[0]):
                        course = str(row.iloc[0]).strip()
                        for i, term in enumerate(terms):
                            if len(row) > i + 1 and pd.notna(row.iloc[i + 1]):
                                course_streams[(course, term)] = int(row.iloc[i + 1])
                            else:
                                course_streams[(course, term)] = 0
                
                st.session_state.courses = course_codes
                st.session_state.professors = staff_names
                st.session_state.terms = terms
                st.session_state.course_preferences = course_preferences
                st.session_state.term_preferences = term_preferences
                st.session_state.professor_term_limits = professor_term_limits
                st.session_state.professor_total_load = professor_total_load
                st.session_state.course_offerings = course_offerings
                st.session_state.course_streams = course_streams
                
                st.success("Excel file loaded successfully!")
                
                if st.button("Next: Data Analysis", type="primary"):
                    st.session_state.step = 2
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")


def show_data_analysis_step():
    """Show data analysis and statistics."""
    st.header("Step 2: Data Analysis & Statistics")
    
    courses = st.session_state.courses
    professors = st.session_state.professors
    terms = st.session_state.terms
    course_preferences = st.session_state.course_preferences
    term_preferences = st.session_state.term_preferences
    professor_term_limits = st.session_state.professor_term_limits
    professor_total_load = st.session_state.professor_total_load
    course_offerings = st.session_state.course_offerings
    course_streams = st.session_state.course_streams
    
    st.subheader("Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Courses", len(courses))
    with col2:
        st.metric("Professors", len(professors))
    with col3:
        offerings_count = sum(course_offerings.values())
        st.metric("Total Offerings", offerings_count)
    with col4:
        total_streams = sum(course_streams.values())
        st.metric("Total Streams", total_streams)
    
    st.subheader("Course Preferences Heatmap (c_ij)")
    st.markdown("**Scale: 0 = Cannot teach, 10 = Strongly prefer**")
    
    course_pref_matrix = pd.DataFrame(index=courses, columns=professors, dtype=float)
    
    for course in courses:
        for prof in professors:
            pref_value = course_preferences.get((course, prof), 0)
            course_pref_matrix.loc[course, prof] = pref_value
    
    fig1 = px.imshow(
        course_pref_matrix.values,
        labels=dict(x="Professor", y="Course", color="Preference Score"),
        x=course_pref_matrix.columns.tolist(),
        y=course_pref_matrix.index.tolist(),
        color_continuous_scale="RdYlGn",
        range_color=[0, 10],
        title="Course Preferences Matrix - All Courses vs All Staff",
        aspect="auto",
        text_auto=True
    )
    
    fig1.update_layout(
        height=max(800, len(courses) * 20),
        width=max(1200, len(professors) * 40),
        font=dict(size=10),
        xaxis=dict(tickangle=90, title="Staff Members"),
        yaxis=dict(
            title="Courses",
            tickmode='array',
            tickvals=list(range(len(courses))),
            ticktext=courses,
            tickfont=dict(size=9)
        )
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_pref = course_pref_matrix.mean().mean()
        st.metric("Average Preference", f"{avg_pref:.2f}")
    with col2:
        high_prefs = (course_pref_matrix >= 8).sum().sum()
        st.metric("High Preferences (≥8)", high_prefs)
    with col3:
        pref_range = f"{course_pref_matrix.min().min():.0f} - {course_pref_matrix.max().max():.0f}"
        st.metric("Preference Range", pref_range)
    
    # Term preferences heatmap
    st.subheader("Term Preferences Heatmap (t_jk)")
    st.markdown("**Scale: 0 = Cannot teach, 10 = Strongly prefer**")
    
    term_pref_matrix = pd.DataFrame(index=professors, columns=terms, dtype=float)
    
    for prof in professors:
        for term in terms:
            term_value = term_preferences.get((prof, term), 0)
            term_pref_matrix.loc[prof, term] = term_value
    
    fig2 = px.imshow(
        term_pref_matrix.values,
        labels=dict(x="Term", y="Professor", color="Preference Score"),
        x=term_pref_matrix.columns.tolist(),
        y=term_pref_matrix.index.tolist(),
        color_continuous_scale="RdYlGn",
        range_color=[0, 10],
        title="Term Preferences Matrix - All Staff vs Terms",
        aspect="auto",
        text_auto=True
    )
    
    fig2.update_layout(
        height=max(600, len(professors) * 25),
        width=800,
        font=dict(size=12),
        xaxis=dict(title="Terms"),
        yaxis=dict(title="Staff Members")
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Upload"):
            st.session_state.step = 1
            st.rerun()
    
    with col2:
        if st.button("Next: Clustering Analysis →", type="primary"):
            st.session_state.step = 3
            st.rerun()


def show_clustering_analysis_step():
    """Show clustering analysis page."""
    st.header("Step 3: Clustering Analysis")
    
    courses = st.session_state.courses
    professors = st.session_state.professors
    terms = st.session_state.terms
    course_preferences = st.session_state.course_preferences
    
    course_pref_matrix = pd.DataFrame(index=courses, columns=professors, dtype=float)
    for course in courses:
        for prof in professors:
            pref_value = course_preferences.get((course, prof), 0)
            course_pref_matrix.loc[course, prof] = pref_value
    
    st.markdown("**Analyze patterns and clusters in course preferences**")
    
    clustering_type = st.radio(
        "Select clustering view:",
        ["Course Clustering (Which courses are similar?)", 
         "Professor Clustering (Which professors have similar preferences?)"],
        horizontal=True
    )
    
    # PCA Analysis
    try:
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        st.subheader("PCA & K-Means Clustering Visualization")
        
        n_clusters = st.selectbox(
            "Select number of clusters (K):",
            options=[2, 3, 4, 5],
            index=1,
            help="Choose how many clusters to identify in the data"
        )
        
        if clustering_type == "Course Clustering (Which courses are similar?)":
            if len(courses) >= 2:
                data_for_pca = course_pref_matrix.values
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data_for_pca)
                
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(data_scaled)
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(data_scaled)
                
                pca_df = pd.DataFrame({
                    'PC1': pca_result[:, 0],
                    'PC2': pca_result[:, 1],
                    'Course': courses,
                    'Cluster': [f'Cluster {i+1}' for i in cluster_labels]
                })
                
                fig_pca = px.scatter(
                    pca_df, x='PC1', y='PC2', color='Cluster', text='Course',
                    title=f'PCA of Courses with {n_clusters} K-Means Clusters',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_pca.update_traces(textposition='top center', marker=dict(size=12))
                fig_pca.update_layout(height=700, showlegend=True)
                st.plotly_chart(fig_pca, use_container_width=True)
                
                # Cluster membership
                st.markdown("**Cluster Membership:**")
                cluster_summary = []
                for i in range(n_clusters):
                    cluster_courses = pca_df[pca_df['Cluster'] == f'Cluster {i+1}']['Course'].tolist()
                    cluster_summary.append({
                        'Cluster': f'Cluster {i+1}',
                        'Size': len(cluster_courses),
                        'Courses': ', '.join(cluster_courses)
                    })
                st.dataframe(pd.DataFrame(cluster_summary), hide_index=True, use_container_width=True)
        
        else:  # Professor clustering
            if len(professors) >= 2:
                data_for_pca = course_pref_matrix.T.values
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data_for_pca)
                
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(data_scaled)
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(data_scaled)
                
                pca_df = pd.DataFrame({
                    'PC1': pca_result[:, 0],
                    'PC2': pca_result[:, 1],
                    'Professor': professors,
                    'Cluster': [f'Cluster {i+1}' for i in cluster_labels]
                })
                
                fig_pca = px.scatter(
                    pca_df, x='PC1', y='PC2', color='Cluster', text='Professor',
                    title=f'PCA of Professors with {n_clusters} K-Means Clusters',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_pca.update_traces(textposition='top center', marker=dict(size=12))
                fig_pca.update_layout(height=700, showlegend=True)
                st.plotly_chart(fig_pca, use_container_width=True)
                
                # Cluster membership
                st.markdown("**Cluster Membership:**")
                cluster_summary = []
                for i in range(n_clusters):
                    cluster_profs = pca_df[pca_df['Cluster'] == f'Cluster {i+1}']['Professor'].tolist()
                    cluster_summary.append({
                        'Cluster': f'Cluster {i+1}',
                        'Size': len(cluster_profs),
                        'Professors': ', '.join(cluster_profs)
                    })
                st.dataframe(pd.DataFrame(cluster_summary), hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
    except ImportError:
        st.info("Install scikit-learn to enable PCA clustering: `pip install scikit-learn`")
    
    # Hierarchical Clustering
    st.subheader("Hierarchical Clustering Dendrograms")
    
    if clustering_type == "Course Clustering (Which courses are similar?)":
        if len(courses) >= 2:
            linkage_matrix = linkage(course_pref_matrix.values, method='ward')
            dendro_data = dendrogram(linkage_matrix, labels=courses, no_plot=True)
            
            fig_dendro = go.Figure()
            icoord = np.array(dendro_data['icoord'])
            dcoord = np.array(dendro_data['dcoord'])
            
            for i in range(len(icoord)):
                fig_dendro.add_trace(go.Scatter(
                    x=icoord[i], y=dcoord[i],
                    mode='lines',
                    line=dict(color='rgb(100,100,100)', width=1),
                    showlegend=False
                ))
            
            fig_dendro.update_layout(
                title="Course Dendrogram",
                xaxis=dict(tickvals=list(range(5, len(courses)*10+5, 10)),
                          ticktext=dendro_data['ivl'], tickangle=90),
                yaxis=dict(title="Distance"),
                height=600
            )
            st.plotly_chart(fig_dendro, use_container_width=True)
    
    else:  # Professor clustering
        if len(professors) >= 2:
            linkage_matrix = linkage(course_pref_matrix.T.values, method='ward')
            dendro_data = dendrogram(linkage_matrix, labels=professors, no_plot=True)
            
            fig_dendro = go.Figure()
            icoord = np.array(dendro_data['icoord'])
            dcoord = np.array(dendro_data['dcoord'])
            
            for i in range(len(icoord)):
                fig_dendro.add_trace(go.Scatter(
                    x=icoord[i], y=dcoord[i],
                    mode='lines',
                    line=dict(color='rgb(100,100,100)', width=1),
                    showlegend=False
                ))
            
            fig_dendro.update_layout(
                title="Professor Dendrogram",
                xaxis=dict(tickvals=list(range(5, len(professors)*10+5, 10)),
                          ticktext=dendro_data['ivl'], tickangle=90),
                yaxis=dict(title="Distance"),
                height=600
            )
            st.plotly_chart(fig_dendro, use_container_width=True)
    
    st.markdown("---")
    
    # Cluster Relationships at the END
    try:
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        st.subheader("Cluster Relationship Analysis")
        st.markdown("**Analyze relationships between professor clusters and course clusters**")
        
        n_clusters_analysis = st.selectbox(
            "Select number of clusters for relationship analysis:",
            options=[2, 3, 4, 5],
            index=1,
            key="cluster_relationship_select"
        )
        
        if len(courses) >= 2 and len(professors) >= 2:
            # Course clustering
            scaler_courses = StandardScaler()
            course_data_scaled = scaler_courses.fit_transform(course_pref_matrix.values)
            kmeans_courses = KMeans(n_clusters=n_clusters_analysis, random_state=42, n_init=10)
            course_cluster_labels = kmeans_courses.fit_predict(course_data_scaled)
            
            # Professor clustering
            scaler_profs = StandardScaler()
            prof_data_scaled = scaler_profs.fit_transform(course_pref_matrix.T.values)
            kmeans_profs = KMeans(n_clusters=n_clusters_analysis, random_state=42, n_init=10)
            prof_cluster_labels = kmeans_profs.fit_predict(prof_data_scaled)
            
            course_to_cluster = {courses[i]: course_cluster_labels[i] for i in range(len(courses))}
            prof_to_cluster = {professors[i]: prof_cluster_labels[i] for i in range(len(professors))}
            
            # Side-by-side PCA
            st.markdown("### Side-by-Side PCA Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                pca_courses = PCA(n_components=2)
                pca_courses_result = pca_courses.fit_transform(course_data_scaled)
                
                pca_courses_df = pd.DataFrame({
                    'PC1': pca_courses_result[:, 0],
                    'PC2': pca_courses_result[:, 1],
                    'Course': courses,
                    'Cluster': [f'C{i+1}' for i in course_cluster_labels]
                })
                
                fig_courses = px.scatter(pca_courses_df, x='PC1', y='PC2', color='Cluster', text='Course',
                                        title=f'Courses - {n_clusters_analysis} Clusters')
                fig_courses.update_layout(height=500)
                st.plotly_chart(fig_courses, use_container_width=True)
            
            with col2:
                pca_profs = PCA(n_components=2)
                pca_profs_result = pca_profs.fit_transform(prof_data_scaled)
                
                pca_profs_df = pd.DataFrame({
                    'PC1': pca_profs_result[:, 0],
                    'PC2': pca_profs_result[:, 1],
                    'Professor': professors,
                    'Cluster': [f'P{i+1}' for i in prof_cluster_labels]
                })
                
                fig_profs = px.scatter(pca_profs_df, x='PC1', y='PC2', color='Cluster', text='Professor',
                                      title=f'Professors - {n_clusters_analysis} Clusters')
                fig_profs.update_layout(height=500)
                st.plotly_chart(fig_profs, use_container_width=True)
            
            # Affinity matrix
            st.markdown("### Cross-Cluster Affinity Matrix")
            
            affinity_matrix = np.zeros((n_clusters_analysis, n_clusters_analysis))
            count_matrix = np.zeros((n_clusters_analysis, n_clusters_analysis))
            
            for i, prof in enumerate(professors):
                prof_cluster = prof_cluster_labels[i]
                for j, course in enumerate(courses):
                    course_cluster = course_cluster_labels[j]
                    pref_score = course_pref_matrix.loc[course, prof]
                    affinity_matrix[prof_cluster, course_cluster] += pref_score
                    count_matrix[prof_cluster, course_cluster] += 1
            
            affinity_matrix = np.divide(affinity_matrix, count_matrix, 
                                       where=count_matrix!=0, 
                                       out=np.zeros_like(affinity_matrix))
            
            fig_affinity = px.imshow(
                affinity_matrix,
                labels=dict(x="Course Cluster", y="Professor Cluster", color="Avg Preference"),
                x=[f'C{i+1}' for i in range(n_clusters_analysis)],
                y=[f'P{i+1}' for i in range(n_clusters_analysis)],
                color_continuous_scale="RdYlGn",
                title="Professor-Course Cluster Affinity",
                text_auto='.2f'
            )
            
            fig_affinity.update_layout(height=400)
            st.plotly_chart(fig_affinity, use_container_width=True)
            
            # Sankey Diagram
            st.markdown("### Professor ↔ Course Cluster Connections (Sankey)")
            st.markdown("**Flow diagram showing relationships between professor and course clusters**")
            
            # Build Sankey data
            source_nodes = []
            target_nodes = []
            values = []
            link_colors = []
            
            edge_threshold = 5.0  # Only show flows with avg preference >= 5
            
            for p_cluster in range(n_clusters_analysis):
                for c_cluster in range(n_clusters_analysis):
                    weight = affinity_matrix[p_cluster, c_cluster]
                    if weight >= edge_threshold:
                        source_nodes.append(p_cluster)  # Professor cluster index
                        target_nodes.append(n_clusters_analysis + c_cluster)  # Course cluster index (offset)
                        values.append(weight)
                        
                        # Color based on preference strength
                        if weight >= 8:
                            link_colors.append('rgba(50, 205, 50, 0.4)')  # Green for high
                        elif weight >= 6.5:
                            link_colors.append('rgba(255, 215, 0, 0.4)')  # Yellow for medium
                        else:
                            link_colors.append('rgba(255, 165, 0, 0.4)')  # Orange for low
            
            # Node labels
            node_labels = [f'Prof Cluster {i+1}' for i in range(n_clusters_analysis)] + \
                         [f'Course Cluster {i+1}' for i in range(n_clusters_analysis)]
            
            # Node colors
            node_colors = ['lightblue'] * n_clusters_analysis + ['lightcoral'] * n_clusters_analysis
            
            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=node_labels,
                    color=node_colors,
                    customdata=[f"<br>Members: {', '.join([p for p, c in prof_to_cluster.items() if c == i])}" 
                               if i < n_clusters_analysis 
                               else f"<br>Members: {', '.join([co for co, cl in course_to_cluster.items() if cl == i-n_clusters_analysis])}"
                               for i in range(len(node_labels))],
                    hovertemplate='%{label}<br>%{customdata}<extra></extra>'
                ),
                link=dict(
                    source=source_nodes,
                    target=target_nodes,
                    value=values,
                    color=link_colors,
                    customdata=[f'Avg Preference: {v:.2f}' for v in values],
                    hovertemplate='%{source.label} → %{target.label}<br>%{customdata}<extra></extra>'
                )
            )])
            
            fig_sankey.update_layout(
                title=f"Professor-Course Cluster Flow (Threshold ≥ {edge_threshold})",
                font=dict(size=12),
                height=600
            )
            
            st.plotly_chart(fig_sankey, use_container_width=True)
            
            st.info(f"📊 Sankey diagram shows flows where average preference ≥ {edge_threshold}. Flow thickness represents preference strength. Green = High (≥8), Yellow = Medium (≥6.5), Orange = Moderate.")
            
            # Summary table
            st.markdown("### Strongest Cluster Relationships")
            relationships = []
            for p_cluster in range(n_clusters_analysis):
                for c_cluster in range(n_clusters_analysis):
                    avg_pref = affinity_matrix[p_cluster, c_cluster]
                    if avg_pref > 0:
                        prof_members = [p for p, c in prof_to_cluster.items() if c == p_cluster]
                        course_members = [co for co, cl in course_to_cluster.items() if cl == c_cluster]
                        relationships.append({
                            'Professor Cluster': f'P{p_cluster+1}',
                            'Course Cluster': f'C{c_cluster+1}',
                            'Avg Preference': f'{avg_pref:.2f}',
                            'Professors': ', '.join(prof_members),
                            'Courses': ', '.join(course_members)
                        })
            
            relationships_df = pd.DataFrame(relationships)
            relationships_df = relationships_df.sort_values('Avg Preference', ascending=False)
            st.dataframe(relationships_df, hide_index=True, use_container_width=True)
        
    except ImportError:
        st.info("Install scikit-learn for full clustering features: `pip install scikit-learn`")
    
    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Data Analysis"):
            st.session_state.step = 2
            st.rerun()
    
    with col2:
        if st.button("Run Optimization →", type="primary"):
            if 'course_offerings' in st.session_state:
                st.session_state.step = 4
            else:
                st.session_state.step = 5
            st.rerun()


def show_results_step():
    """Show optimization results."""
    st.header("Optimization Results")
    
    courses = st.session_state.courses
    professors = st.session_state.professors
    terms = st.session_state.terms
    course_offerings = st.session_state.course_offerings
    course_streams = st.session_state.course_streams
    professor_total_load = st.session_state.professor_total_load
    professor_term_limits = st.session_state.professor_term_limits
    course_preferences = st.session_state.course_preferences
    term_preferences = st.session_state.term_preferences
    
    with st.spinner("Running optimization..."):
        try:
            problem = CourseCoveringProblem(
                courses=courses,
                professors=professors,
                terms=terms,
                course_preferences=course_preferences,
                term_preferences=term_preferences,
                course_streams=course_streams,
                professor_total_load=professor_total_load,
                professor_term_limits=professor_term_limits,
                course_offerings=course_offerings
            )
            
            solution = problem.solve()
            
        except Exception as e:
            st.error(f"Optimization error: {str(e)}")
            return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status", solution['status'])
    with col2:
        if solution['objective_value']:
            st.metric("Objective Value", f"{solution['objective_value']:.1f}")
    with col3:
        st.metric("Unassigned", len(solution.get('unassigned_offerings', [])))
    
    if solution['status'] == 'Optimal':
        st.success("Optimization completed successfully!")
        
        assignments = solution.get('assignments', {})
        
        st.subheader("Course Assignment Matrix")
        
        matrix_data = []
        for course in courses:
            row_data = {'Course': course}
            for professor in professors:
                row_data[professor] = 0
            
            for (assigned_course, term), assigned_professor in assignments.items():
                if assigned_course == course:
                    row_data[assigned_professor] = term
            
            matrix_data.append(row_data)
        
        main_matrix_df = pd.DataFrame(matrix_data)
        main_matrix_df.set_index('Course', inplace=True)
        
        st.dataframe(main_matrix_df, height=min(600, len(courses) * 25), use_container_width=True)
        
        if solution.get('unassigned_offerings'):
            st.warning("Some courses could not be assigned")
    
    elif solution['status'] == 'Infeasible':
        st.error("Problem is infeasible")
        if solution.get('constraint_violations'):
            for violation in solution['constraint_violations']:
                st.write(f"• {violation}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            st.session_state.step = st.session_state.step - 1
            st.rerun()
    with col2:
        if st.button("Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()

